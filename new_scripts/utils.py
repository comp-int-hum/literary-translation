import re

def remove_nikkud(text, keep_end=False):
    # Define the regex pattern
    if keep_end:
        return re.sub(r'[\u0591-\u05AF\u05B0-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C7]', '', text)
    return re.sub(r'[\u0591-\u05C7]', '', text)

# Define special symbols
ATNACH = '\u0591'  # Unicode for Atnach (half-verse separator)
PESUCHA = '׃ פ'     # Pesucha separator
SETUMA = '׃ ס'      # Setuma separator

def old_group_verses(verses, group_type):
    """
    Group verses into half-verses, pesucha, or setuma sections.
    
    Args:
    - verses: list of verse strings
    - group_type: how to group the verses, can be 'half', 'pesucha', or 'setuma'
    
    Returns:
    - A list of grouped verse strings.
    - A list of lists of verse indices in their groups
    """
    
    # the half-verse processing requires different processing as each verse must be processed individually
    if group_type == "verse":
        indices = [[i] for i in range(len(verses))] # [[0], [1], [2]...] bc each group is just one verse
        return [remove_nikkud(v) for v in verses], indices
    
    # Define the splitting logic based on group type
    elif group_type == 'half':
        groups = []
        indices = []
        for idx, verse in enumerate(verses):
            words = verse.split()
            split = None
            for i, w in enumerate(words):
                if ATNACH in w:
                    split = i
            if split != None: # if there are half-verses, add them individually
                groups.append(' '.join(words[:split+1]))
                groups.append(' '.join(words[split+1:]))
                indices.append([f"{idx}a"])
                indices.append([f"{idx}b"])
            else: # otherwise, add the whole verse
                groups.append(verse)
                indices.append([f"{idx}"])

        # some post-processing: remove nikkud and PESUCHA and SETUMA symbols
        groups = [remove_nikkud(g, keep_end=False).replace(PESUCHA, '׃').replace(SETUMA, '׃') for g in groups]
        return groups, indices
    
    ##################################################
    # Concatenate all verses into a single text block
    # I think to make the indexing possible, I need to change how I do this. Need to iterate through so I can
    # keep track of verse delineationgs
    elif group_type in ['pesucha', 'setuma']:
        split_char = PESUCHA if group_type == 'pesucha' else SETUMA
        groups = []
        indices = []
        curr_text = ''
        curr_vs = []
        for idx, verse in enumerate(verses):
            if split_char in verse:
                # add the final verse text in
                curr_text += verse
                curr_vs.append(idx)
                # the group is complete, add it to groups
                groups.append(remove_nikkud(curr_text, keep_end=True).replace(PESUCHA, '׃').replace(SETUMA, '׃'))
                indices.append(curr_vs)
                # reset counters
                curr_text = ''
                curr_vs = []
            else:
                curr_text += verse
                curr_vs.append(idx)
        return groups, indices
    else:
        raise ValueError(f"Unknown group type: {group_type}. Choose from 'verse', 'half_verse', 'pesucha', or 'setuma'.")
    



with open('heb_stopwords.txt', 'rt') as f:
    stops = f.readlines()
    stops = [s.strip() for s in stops]


def hebrew_word_tokenizer(text):
    """
    Tokenizer function to split Hebrew text into words.
    This will capture words and also consider special characters like apostrophes.
    
    :param text: A string of Hebrew text.
    :return: A list of tokenized words.
    """
    # Regular expression to match Hebrew words, including those with apostrophes (׳)
    hebrew_word_pattern = r"\b[\u0590-\u05FF׳]+(?:-[\u0590-\u05FF׳]+)*\b"
    
    # Use re.findall to get all the words matching the pattern
    return re.findall(hebrew_word_pattern, text)
