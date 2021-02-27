import csv


class Person:
    def __init__(self, name, status, income, td, ta):
        self.name = name
        self.status = status
        self.income = income
        self.td_org = td
        self.td = sum(td)
        self.ta_org = ta
        self.ta = sum(ta.values)
        self.check()
        self.writeintocsv()

    def check(self):
        if self.status == "Professional" and ((self.td > self.income * 10) or (self.ta > self.income * 25)):
            raise ValueError("IT Raid Alert")
        elif self.status == "Politician" and (self.td > self.income * 10) and (self.ta > self.income * 10):
            raise ValueError("Disproportionate Assets Alert")
        elif self.status == "Employee" and ((self.td > self.income * 20) or (self.ta > self.income * 20)):
            raise ValueError("Scam Alert")

    def writeintocsv(self):
        with open("person.csv", mode='a') as csvfile:
            fieldnames = ['Name', 'Status', 'Income', 'FD_List', 'Asset_Value_Dict']
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writerow(
                {'Name': self.name, 'Status': self.status, 'Income': self.income, 'FD_List': self.td_org,
                 'Asset_Value_Dict': self.ta_org})

    def readfromcsv(self):
        with open("person.csv", mode='r') as csvfile:
            fieldnames = ['Name', 'Status', 'Income', 'FD_List', 'Asset_Value_Dict']
            csv_reader = csv.DictReader(csvfile, fieldnames=fieldnames)
            for row in csv_reader:
                if row["Name"] == self.name:
                    return row
