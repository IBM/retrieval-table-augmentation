from table_augmentation.table import Table
from typing import Optional


# CONSIDER:
# filter out Google Patents page_title.endswith('Google Patents')
# filter tables that are all numbers
# maybe most tables with more than 20 rows are bad?
# related to forum posts, filter out commit logs:
#   Author   Commit    Message   Labels     Comments    Approvals    Date
# CONSIDER: filter tables that are too repetitive
# CONSIDER: filter out tables without a key column

def filter(tbl: Table) -> Optional[str]:
    if size_filter(tbl):
        return 'size'
    elif filter_calendar(tbl):
        return 'calendar'
    elif filter_by_length(tbl):
        return 'length'
    elif filter_posts(tbl):
        return 'posts'
    elif filter_torrents(tbl):
        return 'torrents'
    else:
        return None
    #return size_filter(tbl) or filter_calendar(tbl) or filter_by_length(tbl) or filter_posts(tbl) or filter_torrents(tbl)


# def


def size_filter(tbl: Table) -> bool:
    """
    remove tables with less than four rows or columns
    :param tbl:
    :return:
    """
    return len(tbl.header) < 4 or len(tbl.rows) < 4


def filter_calendar(tbl: Table) -> bool:
    """
    remove anything that looks like a calendar
    :param tbl:
    :return:
    """
    days = 's m t w t f s'
    months = 'jan feb mar apr may jun jul aug sep oct nov dec'
    h1 = ' '.join(h[:1].lower() for h in tbl.header)
    h3 = ' '.join(h[:3].lower() for h in tbl.header)
    return days in h1 or months in h3


def filter_by_length(tbl: Table) -> bool:
    """
    require none empty title, headers no longer than 40 chars and cells no longer than 80 chars
    :param tbl:
    :return:
    """
    if len(tbl.title) == 0:
        return True
    if any(len(h) > 40 for h in tbl.header):
        return True
    if any(any(len(c) > 80 for c in row) for row in tbl.rows):
        return True
    return False


def filter_posts(tbl: Table) -> bool:
    """
    remove anything that looks like a list of forum posts
    :param tbl:
    :return:
    """
    post_headers = ['board', 'comments', 'commenter', 'created', 'date', 'forum',
                    'last post', 'last poster', 'last post info', 'last reply',
                    'messages', 'replies', 'started by', 'subject', 'thread starter', 'topic',
                    'views', 'when', 'written by']
    norm_header = [h.lower() for h in tbl.header]
    not_post_like = [not h.startswith('post') and (h not in post_headers) for h in norm_header]
    return sum(not_post_like) <= 1


def filter_torrents(tbl: Table) -> bool:
    """
    remove anything that looks like a list of torrents
    :param tbl:
    :return:
    """
    hstr = ' '.join(h.lower() for h in tbl.header)
    return 'seeders leechers' in hstr


