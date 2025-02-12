from zipfile import ZipFile
from json import loads
from shutil import rmtree
from os import path, remove


class Data:
    def __init__(self):
        self.data_path = './TRDataChallenge2023.zip'
        self.data_file = 'TRDataChallenge2023.txt'
        self.dataset = []
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

    def extract_zip(self):
        """
        Extracts JSON data from a ZIP archive and processes them line-by-line.
        """
        if not path.exists(self.data_path):
            print(f"Error: ZIP file not found at {self.data_path}")

        self.docs, self.postures, self.sections = list(), list(), list()
        with ZipFile(self.data_path, 'r').open(self.data_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    json_obj = loads(line)
                    self.docs.append(json_obj['documentId'])
                    self.postures.append(json_obj['postures'])
                    cur_sections = list()
                    for section in json_obj['sections']:
                        headtext = section['headtext']
                        paragraphs = section['paragraphs']
                        cur_sections.append((headtext, paragraphs))
                    self.sections.append(cur_sections)

    def prepare_data(self):
        self.x = []
        self.y = []
        for index in range(len(self.docs)):
            y_datapoints = self.postures[index]
            temp_data = []
            for section in self.sections[index]:
                header = section[0]
                paragraphs = []
                for paragraph in section[1]:
                    paragraphs.append(paragraph)
                paragraphs = '\n'.join(paragraphs)
                temp_data.append(f'Header: {header}\n{paragraphs}')
            x_datapoint = '\n\n'.join(temp_data)
            for y_dp in y_datapoints:
                self.x.append(x_datapoint)
                self.y.append(y_dp)

    def describe(self):
        num_postures = sum([len(posture) for posture in self.postures])
        num_paragraphs = sum([sum([len(section[1]) for section in cur_section])
                              for cur_section in self.sections])
        print("\n--- Dataset Info ---")
        print(f"Number of Documents: {len(self.docs)}")
        print(f"Number of Postures: {num_postures}")
        print(f"Number of Paragraphs: {num_paragraphs}")

    def clean(self):
        temp_dir = './TRDataChallenge2023/'
        temp_file = 'TRDataChallenge2023.zip:Zone.Identifier'
        if path.exists(temp_dir):
            rmtree('./TRDataChallenge2023/')
        if path.exists(temp_file):
            remove(temp_file)
