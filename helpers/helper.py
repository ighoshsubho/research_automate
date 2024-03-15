from langchain.agents import load_tools
import time
from langchain.tools import tool
import praw
from dotenv import load_dotenv
import markdown
import os
import resend

load_dotenv()

resend.api_key = os.getenv("RESEND_API_KEY")

# To load Human in the loop
human_tools = load_tools(["human"])

class BrowserTool:
    @tool("Scrape reddit content")
    def scrape_reddit(max_comments_per_post=7):
        """Useful to scrape a reddit content"""
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent="user-agent", # DebateSuspicious9376
        )
        subreddit = reddit.subreddit("LocalLLaMA")
        scraped_data = []

        for post in subreddit.hot(limit=12):
            post_data = {"title": post.title, "url": post.url, "comments": []}

            try:
                post.comments.replace_more(limit=0)  # Load top-level comments only
                comments = post.comments.list()
                if max_comments_per_post is not None:
                    comments = comments[:7]

                for comment in comments:
                    post_data["comments"].append(comment.body)

                scraped_data.append(post_data)

            except praw.exceptions.APIException as e:
                print(f"API Exception: {e}")
                time.sleep(60)  # Sleep for 1 minute before retrying

        return scraped_data
    
def send_email(content:str):
    html_content = markdown.markdown(content)
    params = {
        "from": "onboarding@resend.dev",
        "to": [os.getenv("EMAIL")],
        "subject": os.getenv("SUBJECT_EMAIL"),
        "html": html_content,
    }
    r = resend.Emails.send(params)
    print(r)
    pass
    
if __name__ == "__main__":
    # tool = BrowserTool()
    # print(tool.scrape_reddit())
    pass