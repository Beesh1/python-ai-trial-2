import requests
from bs4 import BeautifulSoup
import ollama
import re
import json

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "llama3.2"


class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = url
        try:
            response = requests.get(url, timeout=10)  # Add a timeout to prevent long waits
            response.raise_for_status()  # Raise an HTTPError for bad responses
            self.body = response.content
            soup = BeautifulSoup(self.body, 'html.parser')
            self.title = soup.title.string if soup.title else "No title found"
            if soup.body:
                for irrelevant in soup.body(["script", "style", "img", "input"]):
                    irrelevant.decompose()
                self.text = soup.body.get_text(separator="\n", strip=True)
            else:
                self.text = ""
            links = [link.get('href') for link in soup.find_all('a')]
            self.links = [link for link in links if link]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            self.title = "No title found (Error fetching page)"
            self.text = ""
            self.links = []

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"


link_system_prompt = "You are provided with a list of links found on a webpage. \
You are able to decide which of the links would be most relevant to include in a brochure about the company, \
such as links to an About page, or a Company page, or Careers/Jobs pages, and more.\n"
link_system_prompt += "You should respond in JSON as in this example:"
link_system_prompt += """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page": "url": "https://another.full.url/careers"}
    ]
}
"""


def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt


def get_links(url):
    website = Website(url)
    messages = [
        {"role": "system", "content": link_system_prompt},
        {"role": "user", "content": get_links_user_prompt(website)}
    ]
    response = ollama.chat(model=MODEL, messages=messages)

    # Debug the response structure
    print("Response structure:", response)

    try:
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            raw_content = response.message.content

            # Try to parse the raw content directly as JSON
            try:
                result = json.loads(raw_content)
            except json.JSONDecodeError:
                # If parsing fails, extract JSON using regex
                json_match = re.search(r"({.*})", raw_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    result = json.loads(json_content)
                else:
                    raise ValueError("No valid JSON found in the response content.")
        else:
            raise AttributeError("Unexpected response structure. Missing 'message.content' attribute.")
    except Exception as e:
        print(f"Error processing response content: {e}")
        raise

    return result


def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    print("Found links:", links)
    for link in links["links"]:
        if not link["url"].startswith(("http://", "https://")):
            print(f"Skipping non-HTTP link: {link['url']}")
            continue
        try:
            result += f"\n\n{link['type']}\n"
            result += Website(link["url"]).get_contents()
        except Exception as e:
            print(f"Error processing link {link['url']}: {e}")
    return result


system_prompt = (
    "You are a professional marketing assistant tasked with creating a short, polished brochure for a company. "
    "Analyze the provided website content and generate a brochure in Markdown format. "
    "The brochure should include key sections such as 'About Us,' 'Our Mission,' 'Careers,' 'Research,' 'Contact Information,' "
    "and other relevant details that appeal to customers, investors, and recruits. "
    "Do not include technical issues, support content, or unrelated status updates. "
    "Use concise and professional language, and format the brochure with Markdown syntax (e.g., headings, bullet points, and links)."
)


# Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':

# system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
# and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
# Include details of company culture, customers and careers/jobs if you have the information."


def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are creating a brochure for the company: {company_name}.\n"
    user_prompt += "Below is the content extracted from the company's website, organized by relevant sections. "
    user_prompt += "Use this content to generate a Markdown brochure with headings and links, focusing on key sections:\n\n"
    user_prompt += get_all_details(url)
    user_prompt += (
        "\n\nEnsure that the brochure includes only relevant sections such as 'About Us,' 'Our Mission,' "
        "'Careers,' 'Research,' and 'Contact Information.' Exclude technical issues, support, and status updates."
    )
    return user_prompt[:20_000]


def create_brochure(company_name, url):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
    ]
    response = ollama.chat(model=MODEL, messages=messages, stream=True)
    brochure_content = ""
    for chunk in response:
        if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
            brochure_content += chunk.message.content
            print(chunk.message.content, end="")


create_brochure("Anthropic", "https://anthropic.com")
