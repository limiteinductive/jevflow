"""Page titles that say what each page is about and tell it apart from the writer's other pages: Jev checks a title, the local LLM proposes new ones, and Jev picks one."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field, create_model

from flow import decide, feed, generator

TITLE_GATE = 0.5
"""A page is retitled when Jev's P(its title describes it) or P(its title tells it apart from the other pages) is under this."""
NUM_TITLES = 3
TITLE_INSTRUCTIONS = (
    f"The user gives the titles of a writer's other pages and the text of one more page. Reply with {NUM_TITLES} short lowercase titles for that page, one per line, "
    "each 2 to 6 plain words about what that page itself says, so it cannot be mistaken for any of the other pages. Reply with the list only."
)
TITLE_EXAMPLES = [
    (
        "Other titles: x post; standup notes\n\nText: Our CI went from 40 minutes to 4 after we cached the Docker layers. One flag did it.",
        "ci 40 to 4 min\ndocker layer cache\nthe one ci flag",
    ),
    (
        "Other titles: x post; ci 40 to 4 min\n\nText: hot take: the best code review happens before the PR, over coffee.",
        "review over coffee\nreview before the pr\ncoffee beats pr review",
    ),
]

router = APIRouter()


class Page(BaseModel):
    text: str
    title: str
    others: list[str]
    """The titles of the writer's other pages."""


@router.post("/title")
async def title(page: Page) -> dict[str, str | None]:
    """A new title for the page when Jev reads its title as not describing it or not telling it apart from `others`, else None."""
    others = "; ".join(page.others) or "none"
    TitleCheck = create_model(
        "TitleCheck",
        __doc__=f"A writer's page is titled '{page.title}'. Their other pages are titled: {others}.",
        describes=(decide.YesNo, Field(description=f"Does '{page.title}' say what this page is about?")),
        distinct=(decide.YesNo, Field(description=f"Can the writer tell this page from their other pages by its title alone? Other titles: {others}.")),
    )
    check = await decide.run(TitleCheck, f"Text: {page.text}")
    weakest = min(("describes", "distinct"), key=check.get)
    if check[weakest] >= TITLE_GATE:
        feed.act(check, "silent", weakest, component="titles")
        return {"title": None}
    reply = await generator.chat(TITLE_INSTRUCTIONS, TITLE_EXAMPLES, f"Other titles: {others}\n\nText: {page.text}", 40)
    taken = {page.title, *page.others}
    options = [option for option in dict.fromkeys(line.strip(" -*•.").lower() for line in reply.splitlines()) if option and option not in taken][:NUM_TITLES]
    if not options:
        feed.act(check, "dropped", weakest, component="titles")
        return {"title": None}
    TitlePick = create_model(
        "TitlePick",
        __doc__=f"A writer's other pages are titled: {others}.",
        pick=(Literal[tuple(options)], Field(description="Which title best says what this page is about and tells it apart from the other pages?")),
    )
    pick = await decide.run(TitlePick, f"Text: {page.text}")
    feed.act(check, "applied", weakest, component="titles")
    feed.act(pick, "applied", "pick", component="titles")
    return {"title": pick["pick"]}
