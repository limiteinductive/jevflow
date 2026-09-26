"""The mirror's question bank: plain yes/no Jev questions about one sentence.

Each question names one property a reader can see in the sentence. `ai_answer` is the answer
that reads as AI; the stacker's weight for the question is constrained to that direction, so a
reason never points the opposite way to its wording. `yes_reason` and `no_reason` are the
writer-facing reasons shown when that answer pushes the sentence toward AI.
`ask` is what the writing flow asks the writer when the reason fires.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Question:
    name: str
    text: str
    yes_reason: str
    no_reason: str
    ai_answer: Literal["yes", "no"] = "yes"
    ask: str = ""
    """A question to the writer when this reason fires; empty when the reason is a feature only."""


QUESTIONS = [
    Question("reads_ai", "Does the sentence read as written by an AI?", "reads as AI overall", "reads as human overall"),
    Question("reads_human", "Does the sentence read as written by a person?", "reads as human overall", "does not read as a person's writing", "no"),
    Question("generic", "Could the sentence fit many different topics?", "generic phrasing", "specific to its topic", ask="What is specific to your case here?"),
    Question("hedging", "Does the sentence hedge its claim?", "hedging", "direct claim", ask="Can you say it without the hedge?"),
    Question("transition", "Does the sentence start with a transition word such as 'Moreover' or 'Additionally'?", "empty transition", "no stock transition", ask="Does it need the transition word?"),
    Question("buzzwords", "Does the sentence use words like 'delve', 'tapestry', 'vibrant', 'crucial' or 'landscape'?", "stock AI vocabulary", "plain vocabulary", ask="Is there a plainer word you would use?"),
    Question("triad", "Does the sentence list three things in a row?", "list of three", "no list of three", ask="Are all three needed?"),
    Question("summary", "Does the sentence sum up what came before?", "summarizing line", "adds something new", ask="Does this repeat what you already said?"),
    Question("fancy", "Does the sentence use a fancy word where a plain word would do?", "inflated wording", "plain wording", ask="What is the plain way to say this?"),
    Question("abstract", "Is the sentence about an abstract idea?", "abstract, not concrete", "concrete", ask="Can you give an example?"),
    Question("both_sides", "Does the sentence weigh two sides against each other?", "balances both sides", "takes one side", ask="Which side do you take?"),
    Question("importance", "Does the sentence say that something is important or essential?", "asserts importance", "no importance claim", ask="Why does it matter, concretely?"),
    Question("polished", "Is the sentence smooth and polished?", "uniformly polished", "rough edges"),
    Question("moral", "Does the sentence state a lesson or a moral?", "moralizing", "no moral", ask="Does the lesson need spelling out?"),
    Question("not_just", "Does the sentence say 'not just X, but Y' or 'not only X, but also Y'?", "'not just X, but Y' pattern", "no contrast pattern", ask="Can you say it straight?"),
    Question("vague_praise", "Does the sentence praise something in vague terms?", "vague praise", "no vague praise", ask="What exactly is good about it?"),
    Question("names_emotion", "Does the sentence name a feeling, such as 'joy' or 'sadness'?", "names feelings instead of showing them", "no named feelings", ask="Can you show the feeling instead of naming it?"),
    Question("same_shape", "Is the sentence shaped like the sentences around it?", "uniform rhythm", "varied rhythm", ask="Could this sentence be shorter or longer?"),
    Question("specific", "Does the sentence contain a name, number, date or place?", "has concrete details", "no concrete details", "no", ask="What is the number, name or date?"),
    Question("typo", "Does the sentence contain a typo or a grammar slip?", "has a slip", "no slips at all", "no"),
    Question("informal", "Does the sentence use informal or slang words?", "informal words", "no informal words", "no"),
    Question("first_person", "Does the writer talk about their own experience?", "personal experience", "no personal experience", "no", ask="Did you see this yourself?"),
    Question("opinion", "Does the sentence state a blunt opinion?", "blunt opinion", "no stated opinion", "no", ask="What do you think?"),
    Question("humor", "Is the sentence funny or sarcastic?", "humor", "no humor", "no"),
    Question("fragment", "Is the sentence a fragment?", "sentence fragment", "complete, regular sentence", "no"),
    Question("odd_detail", "Does the sentence include an odd or surprising detail?", "surprising detail", "no surprising detail", "no"),
    Question("quote", "Does the sentence quote someone's exact words?", "direct quote", "no direct quote", "no"),
    Question("side_topic", "Does the sentence go off on a side topic?", "digression", "stays on the main line", "no"),
    Question("question", "Is the sentence a question?", "a question", "not a question", "no"),
    Question("dialogue", "Is the sentence part of a conversation between characters?", "dialogue", "no dialogue", "no"),
]
