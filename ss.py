from beautifultable import BeautifulTable
table = BeautifulTable()
#table.column_headers = ["Title", "rank", "gender"]
table.append_row(["Jacob", 1, "boy"])
table.append_row(["Isabella", 1, "girl"])
table.append_row(["Ethan", 2, "boy"])
table.append_row(["Sophia", 2, "girl"])
table.append_row(["Michael", 3, "boy"])
print(table)