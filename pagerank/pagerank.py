import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    ret = {}
    num_links = len(corpus[page])
    num_pages = len(corpus)

    # If a page has links
    if num_links != 0:
        for pages in corpus:
            ret[pages] = (1 - damping_factor) / len(corpus)

        for link in corpus[page]:
            ret[link] = damping_factor / num_links
    else:
        for pages in corpus:
            ret[page] = 1 / len(corpus)

    # Probality of randomly chosing a page and not traversing via a link
    for pages in corpus:
        ret[pages] += (1 - damping_factor) / num_pages

    return ret


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ret = {}
    # Init
    for page in corpus:
        ret[page] = 0.0
    page = random.choice(list(corpus.keys()))

    for i in range(1, n):
        random_page_pick_distribution = transition_model(corpus, page, damping_factor)
        for page in random_page_pick_distribution:
            ret[page] = ((i - 1) * ret[page] + random_page_pick_distribution[page]) / i

        # Weighted Random choice. Second param is probalities from the last step which act as weight
        page = random.choices(list(ret.keys()), list(ret.values()), k=1)[0]

    return ret


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    curr = {}
    threshold = 0.0005
    N = len(corpus)
    count = 0
    # Init
    for page in corpus:
        curr[page] = 1 / N

    while True:
        count = 0
        for key in corpus:
            new = (1 - damping_factor) / N
            sum_link = 0
            for page in corpus:
                if key in corpus[page]:
                    sum_link += curr[page] / len(corpus[page])

            new += (damping_factor * sum_link)

            if abs(new - curr[key]) < threshold:
                count += 1
            curr[key] = new

        if count == N:
            break
    return curr


if __name__ == "__main__":
    main()
