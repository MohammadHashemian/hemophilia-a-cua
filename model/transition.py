from model.utils import probability_at_least_one_event


class ProbabilityBuilder:
    def __init__(self, abr: float, ajbr: float, annual_ltb_prob: float) -> None:
        self.abr = abr
        self.ajbr = ajbr
        self.aebr = abr - ajbr
        self.annual_ltb_prob = annual_ltb_prob
        self.weekly_ltb_prob = annual_ltb_prob / 52

    def get_to_minor_prob(self):
        return probability_at_least_one_event(self.aebr, "annual")

    def get_to_major_prob(self):
        return probability_at_least_one_event(self.ajbr, "annual")

    def get_to_ltb_prob(self):
        return probability_at_least_one_event(self.weekly_ltb_prob, "weekly")

    def get_weekly_ltb_prob(self):
        return self.weekly_ltb_prob

    def get_to_healthy(self):
        total = (
            self.get_to_ltb_prob() + self.get_to_minor_prob() + self.get_to_major_prob()
        )
        return 1 - total
