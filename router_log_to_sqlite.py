import json
import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('log.db')
cursor = conn.cursor()

# Create table if it doesn’t exist
cursor.execute('''CREATE TABLE IF NOT EXISTS log_entries
                  (timestamp TEXT, request_id TEXT, model TEXT, messages_json TEXT, response_text TEXT,
                   completion_tokens INTEGER, prompt_tokens INTEGER, total_tokens INTEGER)''')

# Function to parse a log line
def parse_log_line(line):
    # Check if the line contains "Successful call: "
    if "Successful call: " not in line:
        return None
    # Split into timestamp, level, and message
    parts = line.split(' ', 2)  # Split on first two spaces: timestamp, level, rest
    if len(parts) < 3 or parts[1] != "INFO" or "Successful call: " not in parts[2]:
        return None
    timestamp = parts[0]
    # Extract JSON after "Successful call: "
    json_str = parts[2].split("Successful call: ", 1)[1].strip()
    return timestamp, json_str

# Replace 'path_to_your_log_file.log' with your actual log file path
with open('path_to_your_log_file.log', 'r') as log_file:
    for line in log_file:
        parsed = parse_log_line(line)
        if parsed:
            timestamp, json_str = parsed
            try:
                # Parse JSON
                data = json.loads(json_str)
                # Extract fields
                request_id = data["response"]["id"]
                model = data["response"]["model"]
                messages_json = json.dumps(data["request"]["messages"])  # Serialize messages list
                response_text = data["response"]["choices"][0]["message"]["content"]
                completion_tokens = data["response"]["usage"]["completion_tokens"]
                prompt_tokens = data["response"]["usage"]["prompt_tokens"]
                total_tokens = data["response"]["usage"]["total_tokens"]
                
                # Insert into database
                cursor.execute('''INSERT INTO log_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                               (timestamp, request_id, model, messages_json, response_text,
                                completion_tokens, prompt_tokens, total_tokens))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing line: {line.strip()} - {e}")

# Commit changes and close connection
conn.commit()
conn.close()

print("Log data has been successfully cleaned and stored in 'log.db'.")
