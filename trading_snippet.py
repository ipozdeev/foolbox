class FXPosition(object):
    """

    """

    def __init__(self, currency, position_type, price_data, initial_price,
                 initial_quantity, action=None):

        """

        Parameters
        ----------
        currency
        position_type: str
            'long' or 'short'
        action: str
            'buy', 'sell', 'roll over', 'flip'

        Returns
        -------

        """
        self.currency = currency
        self.position_type = position_type
        self.action = action
        self.price_ask = price_data["price_ask"]
        self.price_bid = price_data["price_ask"]
        self.swap_pts_ask = price_data["swap_ask"]
        self.swap_pts_bid = price_data["swap_bid"]
        self.initial_price = initial_price
        self.initial_quantity = initial_quantity

        self.unrealized_pnl = 0
        self.realized_pnl =0

        self.initial_value = self.initial_price * self.initial_quantity
        self.end_quantity = self.initial_quantity
        self.end_value = self.initial_value
        self.avg_price = initial_price
        self.end_type = position_type

    def buy(self, quantity):
        # If the initial position is long, then buy MOAR
        if self.position_type == "long":
            # Increase the quantity
            self.end_quantity = self.initial_quantity + quantity
            # Compute VWAP
            self.avg_price = \
                (self.initial_price * self.initial_quantity +
                 self.price_ask * quantity) / self.end_quantity
        # If short -- partial close at ask or flip to long
        else:
            self.end_quantity = self.initial_quantity - quantity

            self.realized_pnl = self.realized_pnl - \
                quantity * (self.price_ask - self.initial_price)

    def sell(self, quantity):

        if self.position_type == "long":  # partial close
            self.end_quantity = self.initial_quantity - quantity
            self.end_value = self.end_quantity * self.initial_price

        else:  # short moar
            self.end_quantity = self.initial_quantity + quantity
            self.end_value = self.initial_value + self.price_bid * quantity
            self.avg_price = self.end_value / self.end_quantity

    def roll_over(self):
        if self.position_type == "long":
            self.end_value = self.end_value + \
                             self.end_quantity * self.swap_pts_ask
        else:
            self.end_value = self.end_value + \
                             self.end_quantity * self.swap_pts_bid


    def flip(self, quantity):
        pass


def main():
    pass

if __name__ == '__main__':

    currency = "gbp"
    position_type = "short"
    price_data = {"price_ask": 1.27,
                  "price_bid": 1.23,
                  "swap_ask": 0.02,
                  "swap_bid": 0.01}

    initial_price = 1.25
    initial_quantity = 1

    pos = FXPosition(currency, position_type, price_data, initial_price,
                     initial_quantity)


    print(pos.end_value)








