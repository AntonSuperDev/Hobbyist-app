# hobbyist3

collectible pricer

Front end functionality:

• Entering a category o There is a category text box when an item is being added to a catalogue. o That text box is responsive making suggestions based on text matching as a user types the category of the item being logged. o If there is no existing category tag they use, then their text will become a new category.

• Viewing an item page o When an item in a catalogue is clicked, the item detail page opens. There are two horizontal scrolling panes at the bottom for same and similar items for sale on the internet. o The same items are based on the matches from the daily job described below. They are sorted by price low to high. o The similar items are based on the matches from the daily job described below. They are sorted by price high to low.

• When a member is in an item page there are two horizontal scroll panes popultaed by the two critical jobs below. One is triggered by user, one is scheduled daily to accomplish the same thing (daily pricing of all items).

• Job triggered when new item is added to a user’s catalogue: o When a user enters a new item in the catalogue, a text search (using their description and category entered) is used to search for similar items. o Up to 20 matches are identified on the internet and average price for that item on that day is logged in the database. o Up tp 20 near-matches of similar items based on a search with the category only are also added to the database.

• Daily job run on database: o Daily, every item in the database (items in catalogues) has a job run. o The job pulls the top 20 matches for each item found for sale online. The average price of those is saved in the database. o Same as with user input, up to 20 similar but not matchign items are shown in the similar scroll bar
