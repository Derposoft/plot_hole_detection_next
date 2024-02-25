import math
import requests
import time
from bs4 import BeautifulSoup

DEFAULT_HEADERS = {
    "User-Agent": "bot for research",
    "From": "<your-email>@<your-institution>.edu",
}


def get_story(work_id) -> tuple[str, str, str, str]:
    """
    Returns the title, author, summary, and full work given the id
    """
    url = f"https://archiveofourown.org/works/{str(work_id)}?view_full_work=true"
    r = requests.get(url, headers=DEFAULT_HEADERS)
    soup = BeautifulSoup(r.content, "html.parser")

    title = soup.find("h2", {"class": "title"}).text
    author = soup.find("a", {"rel": "author"}).text
    summary = soup.find("div", {"class": "summary"}).text.strip()
    work = soup.select("p > span")
    work = [x.text.strip() for x in work]
    work = "\n".join(work)
    return title, author, summary, work


def query_stories(page, min_kudos=1000, min_words=10000) -> list[int]:
    """Gets the list of work ids on the given page"""
    assert page >= 1 and min_kudos >= 0 and min_words >= 0
    if page == 1:
        work_search_link = f"https://archiveofourown.org/works/search?work_search%5Bquery%5D=&work_search%5Btitle%5D=&work_search%5Bcreators%5D=&work_search%5Brevised_at%5D=&work_search%5Bcomplete%5D=T&work_search%5Bcrossover%5D=&work_search%5Bsingle_chapter%5D=0&work_search%5Bword_count%5D=%3E{min_words}&work_search%5Blanguage_id%5D=en&work_search%5Bfandom_names%5D=&work_search%5Brating_ids%5D=&work_search%5Barchive_warning_ids%5D%5B%5D=16&work_search%5Bcharacter_names%5D=&work_search%5Brelationship_names%5D=&work_search%5Bfreeform_names%5D=&work_search%5Bhits%5D=&work_search%5Bkudos_count%5D=%3E{min_kudos}&work_search%5Bcomments_count%5D=&work_search%5Bbookmarks_count%5D=&work_search%5Bsort_column%5D=_score&work_search%5Bsort_direction%5D=desc&commit=Search"
    else:
        work_search_link = f"https://archiveofourown.org/works/search?commit=Search&page={page}&work_search%5Barchive_warning_ids%5D%5B%5D=16&work_search%5Bbookmarks_count%5D=&work_search%5Bcharacter_names%5D=&work_search%5Bcomments_count%5D=&work_search%5Bcomplete%5D=T&work_search%5Bcreators%5D=&work_search%5Bcrossover%5D=&work_search%5Bfandom_names%5D=&work_search%5Bfreeform_names%5D=&work_search%5Bhits%5D=&work_search%5Bkudos_count%5D=%26gt%3B{min_kudos}&work_search%5Blanguage_id%5D=en&work_search%5Bquery%5D=&work_search%5Brating_ids%5D=&work_search%5Brelationship_names%5D=&work_search%5Brevised_at%5D=&work_search%5Bsingle_chapter%5D=0&work_search%5Bsort_column%5D=_score&work_search%5Bsort_direction%5D=desc&work_search%5Btitle%5D=&work_search%5Bword_count%5D=%26gt%3B{min_words}"
    search = requests.get(work_search_link, DEFAULT_HEADERS)
    soup = BeautifulSoup(search.content, "html.parser")
    work_ids = soup.select("div > .heading > a")
    work_ids = [x.attrs["href"] for x in work_ids]
    work_ids = [x for x in work_ids if "works/" in x]
    work_ids = [int(x.split("/")[-1]) for x in work_ids]
    return work_ids


N_HITS_PER_PAGE = 20
N_DATAPOINTS = 10
DELAY_SEC = 1

data = []

n_pages = math.ceil(N_DATAPOINTS / N_HITS_PER_PAGE)
for page_idx in range(n_pages):
    page_num = page_idx + 1
    story_ids = query_stories(page_num)
    time.sleep(DELAY_SEC)
    for story_id in story_ids:
        data.append(get_story(story_id))
        if len(data) > N_DATAPOINTS:
            exit(0)
