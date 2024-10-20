from dspy.functional import FunctionalModule, cot, predictor


class Mod(FunctionalModule):
    @predictor
    def hard_question(possible_topics: list[str]) -> str:
        """Write a hard question based on one of the topics. It should be answerable by a number."""

    @cot
    def answer(question: str) -> float:
        pass

    def forward(possible_topics: list[str]):
        q = Mod.hard_question(possible_topics=possible_topics)
        a = Mod.answer(question=q)
        return (q, a)
