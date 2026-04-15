# NLP processing of free filled columns in peribank database for later reassembly/data mining.
# Free filled columns are "details" of conditoins. Medicatons are also free filled and 
# delt with separately 
# Install spacy and download the pretrained pipleline
# using the command "python -m spacy download en_core_web_lg" 
# Note it does 3 iterations, the final being partial matching. 
# Partial matching is fairly aggressive, use the first two for fewer fixes but better fidelity

# Last updated 10 April 2023, Maxim Seferovic, seferovi@bcm.edu
# !/usr/bin/env python3

import pickle, math, os as _os
import pandas as pd
from rapidfuzz import fuzz, process as _rf_process
from multiprocessing import Pool, cpu_count
num_processes = cpu_count()

# ── Resolve data files relative to this script, not the caller's CWD ───────
# Critical when launched by the pipeline orchestrator from a different directory.
_DIR = _os.path.dirname(_os.path.abspath(__file__))

MIN_MED_FREQ = 5   # minimum number of pregnancies a medication must appear in
                   # to be retained as a boolean feature (prevents rare 1-2 occurrence
                   # medications from bloating the output file)


def get_medications():
    with open(_os.path.join(_DIR, "Ensemble_meds.pkl"), "rb") as f:
        return pickle.load(f)


def get_words():
    with open(_os.path.join(_DIR, "Ensemble_words.pkl"), "rb") as f:
        return pickle.load(f)


def remove_excessive_whitespace(string):
    return (' '.join(string.split())).strip()
    
    
def manual_edits(df): # modifies the data itself
    things_to_fix = (('pnv','prenatal vitamin'), ('mag sulfate','magnesium sulfate'), 
        ('asa','acetylsalicylic acid'),('ancef','cefazolin'), ('aspirin','acetylsalicylic acid'),
        ('ohp','hydroxyprogesterone caproate'),('ohp caproate','hydroxyprogesterone caproate'), 
        ('glucophage','metformin'), ('pnc','penicillin')) 
    for item in things_to_fix: 
        df.replace(item[0], item[1], inplace=True)
    return df


def save_changes(spelling_changes): 
    spelling_changes = list(set(spelling_changes)) # removes redundancy
    spelling_changes.sort(key=lambda x: float(x.split(',')[2]), reverse=True) #sorts by score
    spelling_changes.insert(0, 'matched,replaced,Levenshtein_ratio,partial_match')
    with open ("Drug_spelling_matches.csv", 'w') as f: 
        f.write("\n".join(spelling_changes))


def save_non_redundant_meds(df):  #output cleaned non-redundant meds for manual review
    cleaned_meds = set()
    for col in df.columns:
        for meds in df[col].str.split(','):
            cleaned_meds.update(meds)
    with open ("Corrected_drug_names_list.csv", 'w') as f:
        f.write("\n".join(sorted(cleaned_meds)))


def manual_curation(match_list): # modifies the match dictionaries
    #These are fixes for replacements that were problomatic after manual review of "Drug_spelling_matches.csv"
    match_list.extend(["vaseline", "prenatal vitamin"])
    manual_interventions = (("inulin","insulin"),("penicillin g","penicillin"))
    for item in manual_interventions:
        while item[0] in match_list:
            match_list = [item[1] if x == item[0] else x for x in match_list]
    return tuple(match_list)


def fuzzy_replace_meds(chunk, threshold=85):   
    spelling_changes = []
    
    def replace_with_best_match(cell, match_list, partial):
            if not isinstance(cell, str) or not cell.strip():
                return cell
            scorer = fuzz.partial_ratio if partial else fuzz.ratio
            # extractOne batches all comparisons in one C call — 5–10× faster
            # than the previous Python loop over match_list.
            result = _rf_process.extractOne(cell, match_list,
                                            scorer=scorer,
                                            score_cutoff=threshold)
            if result is None:
                return cell
            best_match, best_score, _ = result
            if best_match != cell:
                spelling_changes.append(f"{best_match},{cell},{best_score},{partial}")
            return best_match
    
    #Iterations 1/3, match to medication
    medications = manual_curation(get_medications())
    temp_df = chunk.str.split(',', expand=True)
    for col in temp_df.columns:   
        temp_df[col] = temp_df[col].apply(replace_with_best_match, args=(medications, False))
    temp_series_values = temp_df.apply(lambda x: ','.join(y for y in x.astype(str) if y != 'None' and pd.notna(y)), axis=1)
    temp_series = pd.Series(temp_series_values.values, index=chunk.index, name=chunk.name)
    
    #Iterations 2/3, match to words after spitting to words
    words = manual_curation(get_words())   
    temp_df = temp_series.str.replace(",", " _COMMA_ ").str.split(' ', expand=True)
    for col in temp_df.columns:   
        temp_df[col] = temp_df[col].apply(replace_with_best_match, args=(words, False))
    temp_series_values = temp_df.apply(lambda x: ' '.join(y for y in x.astype(str) if y != 'None' and pd.notna(y)), axis=1)
    temp_series_values = temp_series_values.str.replace(" _COMMA_ ", ",")
    temp_series = pd.Series(temp_series_values.values, index=chunk.index, name=chunk.name)
    
    #Iterations 3/3, PARTIAL match to medications
    #Modded. This is very aggressive, recoded for only specific medications where called for rather than whole med list
    meds_for_partial_match = ("insulin", "fentanyl")
    temp_df = temp_series.str.split(',', expand=True)
    for col in temp_df.columns:   
        temp_df[col] = temp_df[col].apply(replace_with_best_match, args=(meds_for_partial_match, True))
    temp_series_values = temp_df.apply(lambda x: ','.join(y for y in x.astype(str) if y != 'None' and pd.notna(y)), axis=1)
    temp_series = pd.Series(temp_series_values.values, index=chunk.index, name=chunk.name)
    
    #Get a full list of what was changed for review
    num_drugs = chunk[chunk != ''].count()
    num_sp_repl = (temp_series != chunk).sum()
    return temp_series, num_drugs, num_sp_repl, spelling_changes


def split_series(series):
    chunk_size = math.ceil(len(series) / num_processes) # math.ceil just rounds up so no fraction
    return [series[i:i + chunk_size] for i in range(0, len(series), chunk_size)]


def apply_multiprocess(df):
    total_num_drugs = 0
    total_num_sp_repl = 0
    spelling_changes = []
    for col in df.columns: 
        print (f"Starting '{col}'...")
        with Pool(processes=num_processes) as pool:
            chunks = split_series(df[col])
            processed_chunks_and_counts = pool.map(fuzzy_replace_meds, chunks)
            processed_chunks = [chunk for chunk, _, _, _ in processed_chunks_and_counts]
            total_num_drugs += sum(num_drugs for _, num_drugs, _, _ in processed_chunks_and_counts)
            total_num_sp_repl += sum(num_sp_repl for _, _, num_sp_repl, _ in processed_chunks_and_counts)
            for changes in (changes for _, _, _, changes in processed_chunks_and_counts):
                spelling_changes.extend(changes)
                
        df[col] = pd.concat(processed_chunks)
  
    print (f"Total of {total_num_sp_repl} of {total_num_drugs} total drug entries revised for spelling")
    save_changes(spelling_changes)
    return df
    
 
def make_Boolean(series):
    """Convert a comma-separated medication series into a boolean feature DataFrame.

    Vectorised replacement for the old str.contains loop approach.
    The original O(n_unique × n_rows) approach called str.contains once per
    unique medication string for the frequency check and once more for each
    surviving column — potentially billions of string scans at 45 k rows.

    This version uses explode() + value_counts() for frequency (O(n_rows ×
    avg_items)) and get_dummies() + groupby().max() for the boolean matrix —
    no str.contains loops at all.  Semantics are identical: exact post-split
    token matching on the already fuzzy-matched medication strings.
    """
    # Split comma-separated lists; guard against non-list entries (NaN etc.)
    med_lists = series.str.split(',').apply(
        lambda x: [item.strip() for item in (x if isinstance(x, list) else [])])

    # ── Step 1: fast frequency count via explode ────────────────────────────
    exploded = med_lists.explode()                         # one item per row
    valid    = exploded[exploded.str.len().fillna(0) >= 3].dropna()
    freq     = valid.value_counts()
    surviving = set(freq.index[freq >= MIN_MED_FREQ])

    if not surviving:
        return pd.DataFrame(index=series.index)

    # ── Step 2: boolean matrix via get_dummies + groupby (no str.contains) ──
    filtered = valid[valid.isin(surviving)]
    dummies  = pd.get_dummies(filtered)                    # one-hot per item
    bool_df  = dummies.groupby(dummies.index).max()        # one row per pregnancy
    bool_df  = bool_df.reindex(series.index, fill_value=0)
    bool_df.columns = [series.name + '_' + c for c in bool_df.columns]
    return bool_df.astype(bool)


def main(): 
    df = pd.read_csv("PBDBfinal_meds.csv", sep="|", index_col='Pregnancy ID', dtype = str)
    df.fillna('', inplace=True)
    #pregnancy_id = df['Pregnancy ID'] 
    #df.drop('Pregnancy ID', axis=1, inplace=True)
    df = df.applymap(remove_excessive_whitespace)
    df = manual_edits(df)
    df = apply_multiprocess(df)
    save_non_redundant_meds(df)  
    
    with Pool(processes=num_processes) as pool:
        processed_series = pool.map(make_Boolean, [df[col] for col in df.columns])
    dfb = pd.concat(processed_series, axis=1)
    #dfb = pd.concat([pregnancy_id, dfb], axis=1)
    dfb.to_csv("PBDBfinal_meds_dictcorrect_bool.csv", sep='|', index=True)
    #df = pd.concat([pregnancy_id, df], axis=1)
    df.to_csv("PBDBfinal_meds_dictcorrect.csv", sep='|', index=True)


if __name__ == '__main__': 
    main()