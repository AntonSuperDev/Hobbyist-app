Deliverables:
	•	Preferably a web app
	•	Along with being browser accessible, there will be packages for deployment on ios and android app stores (using things like cordova, etc.)
	•	Documentation: all objects, methods, attributes, and database schemas documented in words and mapped to their technical names in architecture diagrams
	•	Deployment instructions to web hosting, ios store, and google play
	•	Instructions on setting up accounts for developers on ios and google play
	•	Assistance with creation of privacy policy passable by ios and google play privacy policies
	•	Assistance with creation of disclaimer on valuation of items
	•	Assistance with setup of in app advertising
	•	Ensure app will comply with ios and google play policies
	•	Other run and operate instructions
	•	Estimate of costs to run and operate application
	•	In addition to technical integration with a third party payment processor, identification and instructions for maintenance of that integration

Requirements:

Back end:

	•	Database schema:
	•	concept table fields: Concept unique id, Concept type (relationship, user, items, descriptions, prices, recommendations, ratings, categories, dates, urls, permissions)
	•	relationships table fields: Concept unique id, Item 1 concept unique id, Item 2 concept unique id, Relationship type (rating of item, date of rating, trigger of rating, value of trigger, )
	•	time range table fields: concept unique id, trigger unique id, value (max, 12mo, 6mo, 3mo, 1mo, 1wk, 1d)
	•	value table fields: concept unique id, value unique id, value (double)
	•	type table fields: concept unique id, type unique id, type (average, standard deviation, median, 1st quartile, 2nd quartile, 3rd quartile, and 4th quartile price values)
	•	threshold table fields: concept unique id, threshold unique id, threshold (integer)
	•	users table fields: Concept unique id, User unique id, User name
	•	names table field: concept unique id, name unique id, username
	•	roles table field: concept unique id, role unique id, role, date start, date end
	•	emails table fields: concept unique id, email unique id, email
	•	thresholds table fields: concept unique id, threshold unique id, threshold (double or integer)
	•	items table fields: Concept unique id, Item unique id
	•	descriptiond table fields: concept unique id, description id, description
	•	prices table fields: Concept unique id, Price unique id, price
	•	categories table fields: Concept unique id, Category unique id, Category name
	•	catalogue table fields: Concept unique id, Catalogue unique id, Catalogue name
	•	dates table fields: Concept unique id, Date unique id, date
	•	url table fields: concept unique id, url unique id, url
	•	permissions table fields: concept unique id, permission unique id, permission
	•	Scripts/methods:
	•	registration jobs
	•	login jobs
	•	daily pricing job: Once a day this job appends ‘value of item’, ‘date of value’, and ‘type of value’, and ’time range of value’ values to the relationships table. This daily job calculates the prior max, 12 mo, 6 mo, 1 mo, 1 week, and 1 day moving average, median, 1st, 2nd, 3rd, and 4th quartile values. New values stored in values, items, and dates tables. Matches for price comparison include a reverse image search using the item image and keywords from the description entered during intake.
	•	item intake job: new items can be added to the catalogue shown in the scroll list includes uploading optional picture
	•	item delete job: remove item from catalogue
	•	manage permissions job: app owner role can see which data and features are available to which role and can change those
	•	manage Categories job: Owner role can see which items are tagged with which categories and can change values.
	•	manage alert threshold job: user can control what delta over what time range will trigger an alert to them. This is controlled in account preferences page.
	•	Others

Front End:

	•	Splash screen: Show screen with logo when opening
	•	Registration page
	•	Login page
	•	Home page opens scrollable list of items in portfolio
	•	Outside scroll list, allows addition of new items
	•	Outside scroll list, in app ads targeted based on catalogue items
	•	Outside scroll list there is access to account and privacy management
	•	Plan change
	•	Cancellation
	•	Switches control different privacy controls at right role level
	•	Push notification preferences
	•	Outside scroll list there is access to browse other catalogues and analytics if user has right role
	•	Outside scroll list, app owner can see and use ‘manage permissions’ and ‘manage categories’ jobs
	•	Clicking scroll list item opens detail page
	•	shows detail (role) including:
	•	item name (all)
	•	item description (all)
	•	date added (all)
	•	1d average price (level 2 and up)
	•	1d median price (level 2 and up)
	•	1d quartile prices (level 2 and up)
	•	all 1d values above with options to show the trend of moving max, 12mo, 6mo, 3mo, 1mo, or 1wk moving values (level 3 and up)
	•	highlight items and values with changes > user threshold managed in account management page (level 3 and up)
	•	graph of distribution of current sale prices online for similar items (level 4 and up)
	•	each dot on the curve links to the current sale url (level 4 and up)
	•	view pie chart of categories represented by catalogue and growth of each segment over times ranges varying from 1wk to 12mo (level 4)
	•	Push notifications of item price changes (level 4)
	•	view others anonymized catalogues and all the analytics outlined above (level 5)
	•	hide your catalogue and data (level 5)
	•	The detail is greyed out with a link to subscribe to a higher level if user role is less than permission level required
