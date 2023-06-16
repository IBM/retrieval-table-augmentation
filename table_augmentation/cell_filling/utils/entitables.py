import numpy as np

from .general import truncate


def normalize_entity(x):
    return x.replace("_", " ").strip()


def is_entity(cell):
    return cell.startswith('[') and cell.endswith(']') and cell.find('|') >= 0 and len(cell[1:-1].split('|')) == 2


def cell_text(cell: str, max_char_cell: int):
    if is_entity(cell):
        page, label = cell[1:-1].split('|')
        page = normalize_entity(page)
        assert '«' not in page
        assert '»' not in page
        # if len(page) > 100:
        #    print(f'Long entity name: {page} ({len(page)})')
        return page  # the point of EntiTables is we predict the page
    return truncate(cell, max_char_cell).replace('«', '<').replace('»', '>')


def process_table(table_id, table, opts):
    header_ents = [is_entity(h) for h in table['title']]
    header = [cell_text(h, opts.max_char_cell) for h in table['title']]

    full_title = [table['pgTitle'], table['secondTitle'], table['caption'], ]
    if full_title[1] == full_title[2]:
        full_title[2] = ""
    full_title = " ".join(full_title).strip()
    title = truncate(full_title, opts.max_char_title)

    rows = [[cell_text(cell, opts.max_char_cell) for cell in row] for row in table['data']]
    assert all([len(row) == len(header) for row in rows])

    all_row_ents = [[is_entity(c) for c in row] for row in table['data']]
    all_row_ents = np.array(all_row_ents, dtype=np.bool)
    entity_column = -1
    if all_row_ents.size > 0:
        for i in range(0, min(3, all_row_ents.shape[1])):  # only allow first 3 columns to be main column
            if np.sum(all_row_ents[:, i]) >= int(all_row_ents.shape[0] * 0.8):
                entity_column = i
                break
    if entity_column >= 0:
        row_ents = all_row_ents[:, entity_column].tolist()
    else:
        row_ents = None

    return {'table_id': table_id, 'title': title, 'header': header, 'rows': rows,
            'key_col_ndx': entity_column, 'row_entities': row_ents, 'header_entities': header_ents}
