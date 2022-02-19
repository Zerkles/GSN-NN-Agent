from mesa import Agent


class Lightpath(Agent):
    def __init__(self, unique_id, model, pos, maker_id, portrayal=None):
        super().__init__(unique_id, model)

        self.pos = pos
        self.maker_id = maker_id
        self.portrayal = portrayal
