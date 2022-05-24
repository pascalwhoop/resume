import requests
import yaml
from pprint import pprint
import json
from types import SimpleNamespace
from datetime import date
import os
from dotenv import load_dotenv
load_dotenv()

token=os.environ['TOKEN']
database_id=os.environ['DB']
headers = {
    "Authorization": "Bearer " + token,
    "Content-Type": "application/json",
    "Notion-Version": "2022-02-22"
}

def main():
    database = read_database(database_id, headers)
    experiences = build_experience_from_database(database)
    resume = yaml.safe_load(open('resume.yaml', 'r'))
    resume['work'] = experiences
    json.dump(resume, open('resume.json', 'w'), indent=2)



def dict_to_sns(d):
    return SimpleNamespace(**d)

def read_database(databaseId, headers):
    readUrl = f"https://api.notion.com/v1/databases/{databaseId}/query"
    res = requests.request("POST", readUrl, headers=headers)

    return json.loads(res.text, object_hook=dict_to_sns)

def dedup(arr):
    return sorted(list(set(arr)))

def read_page(id):
    readUrl = f"https://api.notion.com/v1/pages/{id}/"
    res = requests.get(readUrl, headers=headers)
    return res.json()

def read_property(page, property):
    readUrl = f"https://api.notion.com/v1/pages/{page}/properties/{property}"
    res = requests.get(readUrl, headers=headers)
    return res.json()

def get_roles(item):
    rollup_arr = item.properties.Roles.rollup.array
    return dedup([x.select.name for x in rollup_arr])

def get_technologies(item):
    rollup_arr = [arr.multi_select for arr in item.properties.Technologies.rollup.array]
    return dedup([x.name for sublist in rollup_arr for x in sublist])

def get_countries(item):
    rollup_arr = item.properties.Country.rollup.array
    return dedup([x.select.name for x in rollup_arr])

def get_industries(item):
    rollup_arr = item.properties.__dict__.get('Industries Served').rollup.array
    return dedup([x.select.name for x in rollup_arr])

def get_date_range(item):
    """Returns the from and to date for the entry"""
    pairs = [pair.date for pair in item.properties.Dates.rollup.array]
    starts = [date.fromisoformat(pair.start) for pair in pairs]
    ends = [date.fromisoformat(pair.end) for pair in pairs]
    return (min(starts).isoformat(), max(ends).isoformat())

def get_clients_served(item):
    """Get number of clients served"""
    return item.properties.__dict__.get('Number of Projects').rollup.number

def get_company_name(item):
    """Get number of clients served"""
    return item.properties.Name.title[0].text.content

#def clients_served_txt(item):
#    """Get number of clients served"""
#    roles_objects = item.properties.Clients.rollup.array
#    return [x.select.name for x in roles_objects]

def build_dict_for_item(item):
    item = {
        "roles": get_roles(item),
        "technologies": get_technologies(item),
        "countries": get_countries(item),
        "industries": get_industries(item),
        "start_date": get_date_range(item)[0],
        "end_date": get_date_range(item)[1],
        "clients_count": get_clients_served(item),
        "name": get_company_name(item)
    }
    return item

def build_experience_from_database(database):
    experiences = [build_dict_for_item(item) for item in database.results]
    experiences =  sorted(experiences, key=lambda it: it['start_date'])
    experiences.reverse()
    return experiences


if __name__ == "__main__":
    main()

