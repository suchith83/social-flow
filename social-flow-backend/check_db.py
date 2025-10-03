import sqlite3

conn = sqlite3.connect('test.db')
cursor = conn.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = cursor.fetchall()
print('Tables in test.db:', tables if tables else 'None')
conn.close()
