class CyclicalWarmup():
    """
    Cyclical warmup of the KL term
    """
    def __init__(self, epochs=400, cycle=10, start=0., max_w=1.):
        try:
            assert epochs % cycle == 0
        except:
            raise AssertionError("Number of epochs must be divisible by cycle period.")
        self.epochs = epochs
        self.current_epoch = 0
        self.cycle = cycle
        self.current_cycle = 1
        #self.inc = 1
        self.max_w = max_w
        self.start = start if epochs != 0 else max_w
        self.step = (max_w - start) / cycle if epochs != 0 else 0

    @property
    def is_done(self):
        return self.current_epoch >= self.epochs  

    def __iter__(self):
        return self

    def __next__(self):
        #self.current_epoch += 1

        #if self.current_cycle >= self.cycle:
        #    self.inc = -1
        #elif self.current_cycle <= 1:
        #    self.inc = 1

        # Increasing, first half of cycle
        if (self.current_epoch // self.cycle) % 2 == 0:
            num_steps = (self.current_epoch % self.cycle) + 1

        # Decreasing, second half of cycle
        elif (self.current_epoch // self.cycle) % 2 == 1:
            num_steps = self.cycle - self.current_epoch % self.cycle

        current_weight = self.start + self.step * num_steps
        #current_weight = self.start + self.step * self.current_cycle
        #self.current_cycle += self.inc
        self.current_epoch += 1

        return current_weight

    def __repr__(self):
        s = f"CyclicalWarmup(epochs={self.epochs}, cycle={self.cycle}, start={self.start}, max_w={self.max_w})"
        return s
