import sqlite3

conn = sqlite3.connect('fingerprint.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    phone TEXT,
    email TEXT,
    location TEXT,
    age INTEGER,
    blood_group TEXT,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
conn.close()