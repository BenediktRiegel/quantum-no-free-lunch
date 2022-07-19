class Writer:
    def __init__(self, file_path):
        self.file_path = file_path
        f = open(file_path, 'w')
        f.write('')
        f.close()

    def append(self, text):
        text = str(text)
        with open(self.file_path, 'a') as f:
            f.write(text + '\t')
            f.close()
