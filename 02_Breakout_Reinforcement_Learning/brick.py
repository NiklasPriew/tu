class Bricks:
    def __init__(self, layout):
        self.count_remaining_bricks = layout.sum()
        self.layout = layout