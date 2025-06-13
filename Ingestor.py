import csv

class Ingestor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self._load_data()
    
    def _load_data(self):
        with open(self.file_path, 'r', newline='') as file:
            reader = csv.reader(file)
            title = next(reader) # This line contains the data title 
            header = next(reader)  # Assuming first row is header
            for row in reader:
                # Assuming format: Timestamp (UTC-06:00),Value (TCM)
                if "Data are being provided with the understanding" in row[0]:
                    break
                timestamp = row[0]
                value = float(row[1])
                self.data.append({'Timestamp': timestamp, 'Value': value})
    
    def get_data(self):
        return self.data

# Example usage:
if __name__ == '__main__':
    file_path = 'DataSetExport-Discharge Total.Last-24-Hour-Change-in-Storage@08450800-Instantaneous-TCM-20240622194957.csv'  # Replace with your actual file path
    ingestor = Ingestor(file_path)
    data = ingestor.get_data()
    print(data)
    '''
    for entry in data:
        print(f"Timestamp: {entry['Timestamp']}, Value: {entry['Value']}")'''