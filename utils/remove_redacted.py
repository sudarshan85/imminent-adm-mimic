#!/usr/bin/env python

"""
    This script processes each MIMIC note replacing redacted and obvious information
    with appropriate tokens.
"""
import re
from datetime import datetime

def redacorator(func):
    """
    Decorator for replace functions which passes both the original match and
    lower-case version of it to the replace functions.
    """
    def _replace(match):
        ori = match.group()
        text = match.group().strip().lower()
        return func(text, ori)
    return _replace

"""
    All replace functions take in the original text and a lower-cased version
    of it. If the seeking part of the text is found, a replacement token is
    is returned, if not the original text is returned.
"""
@redacorator
def replace_names(text, ori):
    r = ori
    if 'name' in text:
        r = '<name>'
        if 'last' in text:
            if 'doctor' in text:
                r = '<docln>'
            else:
                r = '<ln>'
        elif 'first' in text:
            if 'doctor' in text:
                r = '<docfn>'
            else:
                r = '<fn>'
        elif 'initials' in text:
            r = '<initial>'
    return r

@redacorator
def replace_places(text, ori):
    r = ori
    if 'hospital' in text:
        r = '<hosp>'
    elif ('company' in text) or ('university/college' in text):
        r = '<work>'
    elif 'location' in text:
        r = '<loc>'
    elif 'country' in text:
        r = '<country>'
    elif 'state' in text:
        r = '<state>'
    elif ('address' in text) or ('po box' in text):
        r = '<addr>'
    return r

@redacorator
def replace_dates(text, ori):
    r = ori
    if re.search(r'\d{4}-\d{0,2}-\d{0,2}', text):
        r = '<date>'
    elif (re.search(r'\d{0,2}-\d{0,2}', text)) or (re.search(r'\d{0,2}\/\d{0,2}', text)) or ('month/day' in text):
        r = '<mmdd>'
    elif 'year' in text or re.search(r'\b\d{4}\b', text):
        r = '<year>'
    elif 'month' in text:
        r = '<month>'
    elif 'holiday' in text:
        r = '<hols>'
    elif 'date range' in text:
        r = '<dtrange>'
    return r

@redacorator
def replace_identifiers(text, ori):
    r = ori
    if ('numeric identifier' in text) or ('pager number' in text):
        r = '<pagerno>'
    elif '(radiology)' in text:
        r = '<radclip>'
    elif 'social security number' in text:
        r = '<ssn>'
    elif 'medical record number' in text:
        r = '<mrno>'
    elif 'age over 90' in text:
        r = '<age90>'
    elif 'serial number' in text:
        r = '<sno>'
    elif 'unit number' in text:
        r = '<unitno>'
    elif 'md number' in text:
        r = '<mdno>'
    elif 'telephone/fax' in text:
        r = '<phno>'
    elif 'provider number' in text:
        r = '<pno>'
    elif 'job number' in text:
        r = '<jobno>'
    elif 'dictator info' in text:
        r = '<dictinfo>'
    elif 'contact info' in text:
        r = '<contact>'
    elif 'attending info' in text:
        r = '<attending>'
    return r

@redacorator
def replace_digits(text, ori):
    r = ori
    if re.search(r'\d\d\d', text):
        r = '<3digit>'
    elif re.search(r'\d\d', text):
        r = '<2digit>'
    elif re.search(r'\d', text):
        r = '<1digit>'
    return r

def replace_redacted(text):
    """
    Function that compiles the redacted pattern and calls all the replace functions
    """
    pat = re.compile(r'\[\*\*(.*?)\*\*\]', re.IGNORECASE)

    # replace name types
    text = pat.sub(replace_names, text)

    # replace place types
    text = pat.sub(replace_places, text)

    # replace person identifier types
    text = pat.sub(replace_identifiers, text)

    # replace date types
    text = pat.sub(replace_dates, text)

    # replace remaining digits
    text = pat.sub(replace_digits, text)
    return text

@redacorator
def replace_time(text, ori):
    """
    Replace times with divided up tokens representing the hour.
    E.g., 8:20 AM is replaced by <forenoon>
    Replace 2-digit redacted information that precedes time identifier with
    a generic token
    E.g., [**84**] AM is replaced by <hour>
    """
    r = ori
    if '**' in text:
        r = '<hour>'
    else:
        try:
        # handle exceptions with custom rules
            f, s = text.split()
            s = 'am' if s[0] == 'a' else 'pm'
            l, r = f.split(':')
            if l == '' or l == '00':
                if r == '':
                    r = str(0).zfill(2)
                l = str(12)
            if int(l) > 12:
                l = str(int(l) % 12)
            f = ':'.join([l, r])
            text = ' '.join([f, s])

            d = datetime.strptime(text, '%I:%M %p')
            if d.hour >= 0 and d.hour < 4:
                r = '<midnight>'
            elif d.hour >= 4 and d.hour < 8:
                r = '<dawn>'
            elif d.hour >= 8 and d.hour < 12:
                r = '<forenoon>'
            elif d.hour >= 12 and d.hour < 16:
                r = '<afternoon>'
            elif d.hour >=16 and d.hour <20:
                r = '<dusk>'
            else:
                r = '<night>'
        except ValueError:
            pass
    return r

def replace_misc(text):
    """
    Replaces certain obvious easy to process items in the notes for helping
    downstream modeling
    """
    # replace different types of "year old" with
    # matches: y.o., y/o, years old. year old, yearold
    text = re.sub(r'-?\byears? ?-?old\b|\by(?:o|r)*[ ./-]*o(?:ld)?\b', ' yo', text, flags=re.IGNORECASE)

    # Does the same thing as above but copied from https://arxiv.org/abs/1808.02622v1
    text = re.sub(r'(\d+)\s*(year\s*old|y.\s*o.|yo|year\s*old|year-old|-year-old|-year old)', r'\1 yo', text, flags=re.IGNORECASE)

    # replaces yr, yr's, yrs with years
    text = re.sub(r'\byr[\'s]*\b', 'years', text, re.IGNORECASE)

    # replace Pt and pt with patient, and IN/OUT/OT PT with patient
    # Note: PT also refers to physical therapy and physical therapist
    text = re.sub(r'\b[P|p]t.?|\b(IN|OU?T) PT\b', 'patient ', text)

    # replace sex with consistant token
    text = re.sub(r'\b(gentlman|male|man|m|M)(?!\S)\b', 'male', text)
    text = re.sub(r'\b(female|woman|f|F)(?!\S)\b', 'female', text)

    # replace time types
    text = re.sub(r'\d{0,2}:\d{0,2} \b[A|P]\.?M\.?\b', replace_time, text, flags=re.IGNORECASE)
    text = re.sub(r'\[\*\*(\d{2})\*\*\] \b[a|p].?m.?\b', replace_time, text, flags=re.IGNORECASE)

    # finally remove leftover redacted stuff (mostly empty)
    text = re.sub(r'\[\*\*(.*?)\*\*\]', '', text, flags=re.IGNORECASE)

    return text

def process_notes(text):
    """
    Master function to processes all the notes
    """
    # replace redacted info with tokens
    text = replace_redacted(text)

    # misc scrubbing
    text = replace_misc(text)
    return text
