import sqlite3

#empty the shapes table
conn = sqlite3.connect('datbase.db')
c = conn.cursor()
c.execute("DELETE FROM shapes")
conn.commit()
conn.close()
