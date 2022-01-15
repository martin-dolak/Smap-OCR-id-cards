import json
import csv

DEFAULT_CSV_SEPARATOR = ";"


class IdCardObject:
    def __init__(self):
        self.documentNo = None
        self.lastName = None
        self.firstName = None
        self.birthDate = None
        self.issueDate = None
        self.expirationDate = None
        self.sex = None

    def convert_to_json(self):
        data = json.dumps(self.__dict__)

        return data

    def convert_to_csv(self, separator=DEFAULT_CSV_SEPARATOR):
        with open('../output/out.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, delimiter=separator, fieldnames=sorted(vars(self)))
            w.writeheader()

            w.writerow({k: getattr(self, k) for k in vars(self)})

    def set_attribute(self, attr, txt):
        setattr(self, attr, txt)

    def reset(self):
        self.__init__()


if __name__ == '__main__':
    a = IdCardObject()
    a.convert_to_json()
