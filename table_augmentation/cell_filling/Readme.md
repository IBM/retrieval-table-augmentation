## Overview
different from row or cell population, in cell filling we make the training and test queries up front.
We use make_queries.py for this except for the EntiTables test set which uses the entitables_test_queries.py script.

## Passages
for both EntiTables and WebTables there are specific cell filling passage dump scripts. 
These can be removed in favor of tables2passages.py pretty easily.


## TODOs
* augmentation_tasks.make_cell_query is used only for 
* Need to use the make_queries.py logic to create Tables with answers - not queries directly
* related to above, the Table.answers seems to not be used. 
