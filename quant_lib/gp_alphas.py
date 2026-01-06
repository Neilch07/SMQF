from .factor import factor
import numpy as np
import pandas as pd
from .analysis import *


class gp_alpha_001(factor):
    def set_factor_property(self):
        """
        GP Alpha#1: Genetic Programming Generated Factor
        Formula: add(
            mul(rank(diff5(diff5(add(closes, opens)))), ts_min5(lows)),
            diff5(diff5(mul(diff5(lows), highs)))
        )

        Performance Metrics:
        - Fitness: 7.4493
        - IR: 1.0554
        - Net IR: 0.0483
        - t-stat: 0.9425
        - Max Drawdown: 6.6919
        """
        factor_property = {
            "factor_name": "gp_alpha_001",
            "data_needed": ["close", "open", "high", "low"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        closes = self.data["close"]
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]

        # Part 1: mul(rank(diff5(diff5(add(closes, opens)))), ts_min5(lows))
        sum_co = closes + opens
        diff5_1 = ts_delta(sum_co, 5)
        diff5_2 = ts_delta(diff5_1, 5)
        rank_part = cs_rank(diff5_2)
        ts_min5_lows = ts_min(lows, 5)
        part1 = rank_part * ts_min5_lows

        # Part 2: diff5(diff5(mul(diff5(lows), highs)))
        diff5_lows = ts_delta(lows, 5)
        mul_part = diff5_lows * highs
        diff5_3 = ts_delta(mul_part, 5)
        part2 = ts_delta(diff5_3, 5)

        # Final factor: add(part1, part2)
        self.factor_data = -part1 - part2


class gp_alpha_002(factor):
    def set_factor_property(self):
        """
        GP Alpha#2: Genetic Programming Generated Factor
        Formula: add(
            div(add(rank(closes), sub(add(sub(highs, -4), rank(sub(amounts, -2))),
            ts_min5(add(ts_min5(ts_max5(lows)), neg(opens))))), lows),
            diff5(add(-2, opens))
        )

        Performance Metrics:
        - Fitness: 4.0032
        - IR: 0.8465
        - Net IR: 0.0698
        - t-stat: 0.7598
        """
        factor_property = {
            "factor_name": "gp_alpha_002",
            "data_needed": ["close", "open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        closes = self.data["close"]
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Numerator part
        rank_closes = cs_rank(closes)
        sub1 = highs - (-4)  # highs + 4
        sub2 = amounts - (-2)  # amounts + 2
        rank_sub2 = cs_rank(sub2)
        add1 = sub1 + rank_sub2

        ts_max5_lows = ts_max(lows, 5)
        ts_min5_max = ts_min(ts_max5_lows, 5)
        neg_opens = -opens
        add2 = ts_min5_max + neg_opens
        ts_min5_add2 = ts_min(add2, 5)

        sub3 = add1 - ts_min5_add2
        numerator = rank_closes + sub3

        # Division by lows
        part1 = numerator / lows.where(lows != 0, np.nan)

        # Part 2: diff5(add(-2, opens))
        add3 = -2 + opens
        part2 = ts_delta(add3, 5)

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_003(factor):
    def set_factor_property(self):
        """
        GP Alpha#3: Genetic Programming Generated Factor
        Formula: add(
            div(opens, rank(ts_ma5(ts_min5(lows)))),
            sub(neg(6), rank(lows))
        )

        Performance Metrics:
        - Fitness: 3.8809
        - IR: 0.5706
        - Net IR: 0.0531
        - t-stat: 0.5402
        """
        factor_property = {
            "factor_name": "gp_alpha_003",
            "data_needed": ["open", "low"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        lows = self.data["low"]

        # Part 1: div(opens, rank(ts_ma5(ts_min5(lows))))
        ts_min5_lows = ts_min(lows, 5)
        ts_ma5_min = ts_mean(ts_min5_lows, 5)
        rank_ma = cs_rank(ts_ma5_min)
        part1 = opens / rank_ma.where(rank_ma != 0, np.nan)

        # Part 2: sub(neg(6), rank(lows))
        neg_6 = -6
        rank_lows = cs_rank(lows)
        part2 = neg_6 - rank_lows

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_004(factor):
    def set_factor_property(self):
        """
        GP Alpha#4: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, opens), add(closes, neg(opens))))), mul(lows, lows)))), closes)
        )

        Performance Metrics (ID: 0):
        - Fitness: 8.0753, IR: 1.2354, Net IR: 0.0320, t-stat: 1.1107, Max DD: 7.3184
        """
        factor_property = {
            "factor_name": "gp_alpha_004",
            "data_needed": ["close", "open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        closes = self.data["close"]
        opens = self.data["open"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2: add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, opens), add(closes, neg(opens))))), mul(lows, lows)))), closes)
        mul_ao = amounts * opens
        add_c_no = closes + (-opens)
        div_part = mul_ao / add_c_no.where(add_c_no != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + closes

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_005(factor):
    def set_factor_property(self):
        """
        GP Alpha#5: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(mul(closes, amounts), add(closes, neg(opens))))), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 7):
        - Fitness: 7.5844, IR: 1.1621, Net IR: 0.0320, t-stat: 1.0365, Max DD: 6.8167
        """
        factor_property = {
            "factor_name": "gp_alpha_005",
            "data_needed": ["close", "open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        closes = self.data["close"]
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2: add(diff5(diff5(add(rank(ts_ma5(div(mul(closes, amounts), add(closes, neg(opens))))), mul(lows, lows)))), highs)
        mul_ca = closes * amounts
        add_c_no = closes + (-opens)
        div_part = mul_ca / add_c_no.where(add_c_no != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_006(factor):
    def set_factor_property(self):
        """
        GP Alpha#6: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, amounts), add(closes, neg(opens))))), mul(lows, lows)))), closes)
        )

        Performance Metrics (ID: 8-10):
        - Fitness: 7.5134, IR: 1.1927, Net IR: 0.0379, t-stat: 1.0715, Max DD: 6.7859
        """
        factor_property = {
            "factor_name": "gp_alpha_006",
            "data_needed": ["close", "open", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        closes = self.data["close"]
        opens = self.data["open"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2: add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, amounts), add(closes, neg(opens))))), mul(lows, lows)))), closes)
        mul_aa = amounts * amounts
        add_c_no = closes + (-opens)
        div_part = mul_aa / add_c_no.where(add_c_no != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + closes

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_007(factor):
    def set_factor_property(self):
        """
        GP Alpha#7: Genetic Programming Generated Factor
        Formula: add(
            neg(lows),
            diff5(diff5(mul(add(-7, opens), lows)))
        )

        Performance Metrics (ID: 11-14):
        - Fitness: 7.4423, IR: 1.2912, Net IR: 0.0393, t-stat: 1.1860, Max DD: 6.8420
        """
        factor_property = {
            "factor_name": "gp_alpha_007",
            "data_needed": ["open", "low"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        lows = self.data["low"]

        # Part 1: neg(lows)
        part1 = -lows

        # Part 2: diff5(diff5(mul(add(-7, opens), lows)))
        add_7o = -7 + opens
        mul_part = add_7o * lows
        diff5_1 = ts_delta(mul_part, 5)
        part2 = ts_delta(diff5_1, 5)

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_008(factor):
    def set_factor_property(self):
        """
        GP Alpha#8: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, amounts), add(closes, neg(opens))))), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 15-26):
        - Fitness: 7.4342, IR: 1.1793, Net IR: 0.0379, t-stat: 1.0616, Max DD: 6.7316
        """
        factor_property = {
            "factor_name": "gp_alpha_008",
            "data_needed": ["close", "open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        closes = self.data["close"]
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2: add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, amounts), add(closes, neg(opens))))), mul(lows, lows)))), highs)
        mul_aa = amounts * amounts
        add_c_no = closes + (-opens)
        div_part = mul_aa / add_c_no.where(add_c_no != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_009(factor):
    def set_factor_property(self):
        """
        GP Alpha#9: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, amounts), div(closes, neg(opens))))), mul(lows, lows)))), lows)
        )

        Performance Metrics (ID: 27):
        - Fitness: 7.4086, IR: 1.1194, Net IR: 0.0320, t-stat: 0.9982, Max DD: 6.6522
        """
        factor_property = {
            "factor_name": "gp_alpha_009",
            "data_needed": ["close", "open", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        closes = self.data["close"]
        opens = self.data["open"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2: add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, amounts), div(closes, neg(opens))))), mul(lows, lows)))), lows)
        mul_aa = amounts * amounts
        div_c_no = closes / (-opens).where(opens != 0, np.nan)
        div_part = mul_aa / div_c_no.where(div_c_no != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + lows

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_010(factor):
    def set_factor_property(self):
        """
        GP Alpha#10: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, diff5(opens)), sub(closes, neg(amounts))))), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 28):
        - Fitness: 7.2458, IR: 1.0532, Net IR: 0.0320, t-stat: 0.9304, Max DD: 6.4174
        """
        factor_property = {
            "factor_name": "gp_alpha_010",
            "data_needed": ["close", "open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        closes = self.data["close"]
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2
        diff5_opens = ts_delta(opens, 5)
        mul_ad5o = amounts * diff5_opens
        sub_c_na = closes - (-amounts)
        div_part = mul_ad5o / sub_c_na.where(sub_c_na != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_011(factor):
    def set_factor_property(self):
        """
        GP Alpha#11: Genetic Programming Generated Factor
        Formula: add(
            add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, diff5(neg(neg(opens)))), lows))), mul(lows, lows)))), highs),
            diff5(diff5(mul(opens, lows)))
        )

        Performance Metrics (ID: 29):
        - Fitness: 7.2337, IR: 1.0670, Net IR: 0.0320, t-stat: 0.9406, Max DD: 6.3991
        """
        factor_property = {
            "factor_name": "gp_alpha_011",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1
        diff5_nno = ts_delta(-(-opens), 5)  # This simplifies to ts_delta(opens, 5)
        mul_ad5o = amounts * diff5_nno
        div_part = mul_ad5o / lows.where(lows != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_1 = ts_delta(add_inner, 5)
        diff5_2 = ts_delta(diff5_1, 5)
        part1 = diff5_2 + highs

        # Part 2: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_3 = ts_delta(mul_ol, 5)
        part2 = ts_delta(diff5_3, 5)

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_012(factor):
    def set_factor_property(self):
        """
        GP Alpha#12: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(rank(div(div(9, ts_min5(ts_max5(opens))), amounts))), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 30):
        - Fitness: 7.2167, IR: 1.0965, Net IR: 0.0320, t-stat: 0.9753, Max DD: 6.4566
        """
        factor_property = {
            "factor_name": "gp_alpha_012",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2
        ts_max5_o = ts_max(opens, 5)
        ts_min5_max = ts_min(ts_max5_o, 5)
        div_9_tsmin = 9 / ts_min5_max.where(ts_min5_max != 0, np.nan)
        div_part = div_9_tsmin / amounts.where(amounts != 0, np.nan)
        rank_1 = cs_rank(div_part)
        rank_2 = cs_rank(rank_1)
        mul_ll = lows * lows
        add_inner = rank_2 + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_013(factor):
    def set_factor_property(self):
        """
        GP Alpha#13: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(add(-7, opens), lows))),
            neg(opens)
        )

        Performance Metrics (ID: 31-32):
        - Fitness: 7.2001, IR: 1.2621, Net IR: 0.0393, t-stat: 1.1618, Max DD: 6.6281
        """
        factor_property = {
            "factor_name": "gp_alpha_013",
            "data_needed": ["open", "low"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        lows = self.data["low"]

        # Part 1: diff5(diff5(mul(add(-7, opens), lows)))
        add_7o = -7 + opens
        mul_part = add_7o * lows
        diff5_1 = ts_delta(mul_part, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2: neg(opens)
        part2 = -opens

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_014(factor):
    def set_factor_property(self):
        """
        GP Alpha#14: Genetic Programming Generated Factor
        Formula: add(
            add(diff5(diff5(add(rank(ts_ma5(div(mul(amounts, amounts), add(closes, neg(opens))))), mul(lows, opens)))), highs),
            diff5(diff5(mul(opens, lows)))
        )

        Performance Metrics (ID: 33):
        - Fitness: 7.1737, IR: 1.1680, Net IR: 0.0396, t-stat: 1.0422, Max DD: 6.4339
        """
        factor_property = {
            "factor_name": "gp_alpha_014",
            "data_needed": ["close", "open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        closes = self.data["close"]
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1
        mul_aa = amounts * amounts
        add_c_no = closes + (-opens)
        div_part = mul_aa / add_c_no.where(add_c_no != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_lo = lows * opens
        add_inner = rank_part + mul_lo
        diff5_1 = ts_delta(add_inner, 5)
        diff5_2 = ts_delta(diff5_1, 5)
        part1 = diff5_2 + highs

        # Part 2: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_3 = ts_delta(mul_ol, 5)
        part2 = ts_delta(diff5_3, 5)

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_015(factor):
    def set_factor_property(self):
        """
        GP Alpha#15: Genetic Programming Generated Factor
        Formula: add(
            add(diff5(diff5(add(rank(ts_ma5(div(div(10, amounts), neg(ts_max5(opens))))), mul(lows, lows)))), highs),
            diff5(diff5(mul(opens, add(2, lows))))
        )

        Performance Metrics (ID: 34-36):
        - Fitness: 7.1616, IR: 1.1789, Net IR: 0.0396, t-stat: 1.0577, Max DD: 6.4648
        """
        factor_property = {
            "factor_name": "gp_alpha_015",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1
        div_10a = 10 / amounts.where(amounts != 0, np.nan)
        ts_max5_o = ts_max(opens, 5)
        neg_tsmax = -ts_max5_o
        div_part = div_10a / neg_tsmax.where(neg_tsmax != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_1 = ts_delta(add_inner, 5)
        diff5_2 = ts_delta(diff5_1, 5)
        part1 = diff5_2 + highs

        # Part 2: diff5(diff5(mul(opens, add(2, lows))))
        add_2l = 2 + lows
        mul_ol = opens * add_2l
        diff5_3 = ts_delta(mul_ol, 5)
        part2 = ts_delta(diff5_3, 5)

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_016(factor):
    def set_factor_property(self):
        """
        GP Alpha#16: Genetic Programming Generated Factor
        Formula: add(
            add(diff5(diff5(add(rank(ts_ma5(div(-9, div(amounts, lows)))), mul(lows, lows)))), highs),
            diff5(diff5(mul(opens, lows)))
        )

        Performance Metrics (ID: 37-41):
        - Fitness: 7.0640, IR: 1.1603, Net IR: 0.0379, t-stat: 1.0426, Max DD: 6.3765
        """
        factor_property = {
            "factor_name": "gp_alpha_016",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1
        div_al = amounts / lows.where(lows != 0, np.nan)
        div_part = -9 / div_al.where(div_al != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_1 = ts_delta(add_inner, 5)
        diff5_2 = ts_delta(diff5_1, 5)
        part1 = diff5_2 + highs

        # Part 2: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_3 = ts_delta(mul_ol, 5)
        part2 = ts_delta(diff5_3, 5)

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_017(factor):
    def set_factor_property(self):
        """
        GP Alpha#17: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, add(2, lows)))),
            add(diff5(diff5(add(rank(ts_ma5(div(-9, amounts))), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 42):
        - Fitness: 7.0571, IR: 1.1509, Net IR: 0.0379, t-stat: 1.0288, Max DD: 6.3331
        """
        factor_property = {
            "factor_name": "gp_alpha_017",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, add(2, lows))))
        add_2l = 2 + lows
        mul_ol = opens * add_2l
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2
        div_part = -9 / amounts.where(amounts != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_018(factor):
    def set_factor_property(self):
        """
        GP Alpha#18: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, add(2, lows)))),
            add(diff5(diff5(add(rank(ts_ma5(amounts)), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 43):
        - Fitness: 7.0380, IR: 1.1500, Net IR: 0.0379, t-stat: 1.0315, Max DD: 6.3332
        """
        factor_property = {
            "factor_name": "gp_alpha_018",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, add(2, lows))))
        add_2l = 2 + lows
        mul_ol = opens * add_2l
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2
        ts_ma5_part = ts_mean(amounts, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_019(factor):
    def set_factor_property(self):
        """
        GP Alpha#19: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(div(10, amounts), neg(lows)))), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 44):
        - Fitness: 7.0077, IR: 1.1562, Net IR: 0.0396, t-stat: 1.0377, Max DD: 6.3218
        """
        factor_property = {
            "factor_name": "gp_alpha_019",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2
        div_10a = 10 / amounts.where(amounts != 0, np.nan)
        neg_lows = -lows
        div_part = div_10a / neg_lows.where(neg_lows != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_020(factor):
    def set_factor_property(self):
        """
        GP Alpha#20: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(-9, amounts))), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 45):
        - Fitness: 6.9836, IR: 1.1489, Net IR: 0.0379, t-stat: 1.0321, Max DD: 6.3037
        """
        factor_property = {
            "factor_name": "gp_alpha_020",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2
        div_part = -9 / amounts.where(amounts != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_021(factor):
    def set_factor_property(self):
        """
        GP Alpha#21: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(-3, amounts))), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 46):
        - Fitness: 6.9836, IR: 1.1489, Net IR: 0.0379, t-stat: 1.0321, Max DD: 6.3037
        """
        factor_property = {
            "factor_name": "gp_alpha_021",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2
        div_part = -3 / amounts.where(amounts != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_022(factor):
    def set_factor_property(self):
        """
        GP Alpha#22: Genetic Programming Generated Factor
        Formula: add(
            diff5(diff5(mul(opens, lows))),
            add(diff5(diff5(add(rank(ts_ma5(div(-1, amounts))), mul(lows, lows)))), highs)
        )

        Performance Metrics (ID: 47):
        - Fitness: 6.9836, IR: 1.1489, Net IR: 0.0379, t-stat: 1.0321, Max DD: 6.3037
        """
        factor_property = {
            "factor_name": "gp_alpha_022",
            "data_needed": ["open", "high", "low", "amount"],
            "factor_type": "market",
        }
        return factor_property

    def calc_factor(self):
        opens = self.data["open"]
        highs = self.data["high"]
        lows = self.data["low"]
        amounts = self.data["amount"]

        # Part 1: diff5(diff5(mul(opens, lows)))
        mul_ol = opens * lows
        diff5_1 = ts_delta(mul_ol, 5)
        part1 = ts_delta(diff5_1, 5)

        # Part 2
        div_part = -1 / amounts.where(amounts != 0, np.nan)
        ts_ma5_part = ts_mean(div_part, 5)
        rank_part = cs_rank(ts_ma5_part)
        mul_ll = lows * lows
        add_inner = rank_part + mul_ll
        diff5_2 = ts_delta(add_inner, 5)
        diff5_3 = ts_delta(diff5_2, 5)
        part2 = diff5_3 + highs

        # Final factor
        self.factor_data = part1 + part2


class gp_alpha_023(factor):
    def set_factor_property(self):
        """
        GP Alpha#23: Genetic Programming Generated Factor
        Formula: div(
            rank(add(rank(add(highs, add(add(closes, closes), closes))),
                add(amounts, add(add(add(amounts, add(add(closes, highs), closes)), add(highs, closes)),
                add(add(amounts, add(rank(closes), add(closes, add(mul(add(amounts, opens), div(lows, opens)), closes)))), closes))))),
            rank(rank(add(rank(closes),
                add(amounts, add(add(rank(closes), add(add(add(lows, add(highs, closes)), ts_ma5(amounts)),
                add(add(amounts, add(add(amounts, add(highs, closes)), closes)), closes))),
                add(highs, add(closes, neg(sub(opens, lows)))))))))
        )

        Performance Metrics (ID: 0):
        - Fitness: 1377.5275, IR: 2.6952, Net IR: -0.0000, t-stat: 2.5494, Max DD: 1303.0078
        """
        factor_property = {
            'factor_name': 'gp_alpha_023',
            'data_needed': ['close', 'open', 'high', 'low', 'amount'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        amounts = self.data['amount']

        # Numerator: rank(add(rank(add(highs, add(add(closes, closes), closes))), ...))
        # Inner part: add(highs, add(add(closes, closes), closes))
        inner1 = highs + (closes + closes + closes)
        rank_inner1 = cs_rank(inner1)

        # Complex nested additions for numerator
        part_a1 = amounts + ((closes + highs) + closes)
        part_a2 = amounts + part_a1 + highs + closes

        # mul(add(amounts, opens), div(lows, opens))
        mul_part = (amounts + opens) * (lows / opens.where(opens != 0, np.nan))

        part_a3 = amounts + cs_rank(closes) + closes + mul_part + closes
        part_a = part_a2 + part_a3 + closes

        numerator_inner = rank_inner1 + amounts + part_a
        numerator = cs_rank(numerator_inner)

        # Denominator: rank(rank(add(rank(closes), ...)))
        rank_closes = cs_rank(closes)

        # Complex nested structure for denominator
        part_b1 = lows + highs + closes
        part_b2 = ts_mean(amounts, 5)
        part_b3 = amounts + (amounts + (highs + closes) + closes) + closes

        part_b = cs_rank(closes) + (part_b1 + part_b2 + part_b3) + highs + closes - (opens - lows)

        denominator_inner = rank_closes + amounts + (cs_rank(closes) + part_b)
        denominator = cs_rank(cs_rank(denominator_inner))

        # Final factor: div(numerator, denominator)
        self.factor_data = numerator / denominator.where(denominator != 0, np.nan)


class gp_alpha_024(factor):
    def set_factor_property(self):
        """
        GP Alpha#24: Simplified version with highs only
        Formula: div(
            rank(add(rank(add(highs, add(add(closes, closes), closes))), highs)),
            rank(rank(add(rank(closes),
                add(amounts, add(add(rank(closes), add(add(add(lows, add(highs, amounts)), ts_ma5(diff5(rank(lows)))),
                add(add(amounts, add(amounts, highs)), closes))), add(highs, add(closes, neg(sub(opens, lows)))))))))
        )

        Performance Metrics (ID: 3):
        - Fitness: 1377.5275, IR: 2.6952, Net IR: -0.0000, t-stat: 2.5494, Max DD: 1303.0078
        """
        factor_property = {
            'factor_name': 'gp_alpha_024',
            'data_needed': ['close', 'open', 'high', 'low', 'amount'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        amounts = self.data['amount']

        # Numerator: simpler structure
        inner1 = highs + (closes + closes + closes)
        rank_inner1 = cs_rank(inner1)
        numerator_inner = rank_inner1 + highs
        numerator = cs_rank(numerator_inner)

        # Denominator
        rank_closes = cs_rank(closes)

        # ts_ma5(diff5(rank(lows)))
        rank_lows = cs_rank(lows)
        diff5_rank_lows = ts_delta(rank_lows, 5)
        ts_ma5_part = ts_mean(diff5_rank_lows, 5)

        part_b1 = lows + highs + amounts
        part_b2 = amounts + amounts + highs + closes
        part_b = cs_rank(closes) + (part_b1 + ts_ma5_part + part_b2) + highs + closes - (opens - lows)

        denominator_inner = rank_closes + amounts + (cs_rank(closes) + part_b)
        denominator = cs_rank(cs_rank(denominator_inner))

        self.factor_data = numerator / denominator.where(denominator != 0, np.nan)


class gp_alpha_025(factor):
    def set_factor_property(self):
        """
        GP Alpha#25: With nested closures and amounts
        Formula: Complex nested structure with multiple closures

        Performance Metrics (ID: 12):
        - Fitness: 1377.5275, IR: 2.6952, Net IR: -0.0000, t-stat: 2.5494, Max DD: 1303.0078
        """
        factor_property = {
            'factor_name': 'gp_alpha_025',
            'data_needed': ['close', 'open', 'high', 'low', 'amount'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        amounts = self.data['amount']

        # Numerator
        inner1 = highs + (closes + closes + closes)
        rank_inner1 = cs_rank(inner1)

        part_a = amounts + cs_rank(closes) + amounts + closes
        part_b = amounts + (closes + highs) + closes + highs + closes
        part_c = amounts + part_a + closes

        numerator_inner = rank_inner1 + amounts + part_b + part_c
        numerator = cs_rank(numerator_inner)

        # Denominator
        rank_closes = cs_rank(closes)
        part_d1 = lows + highs + closes
        part_d2 = ts_mean(amounts, 5)
        part_d3 = amounts + (amounts + (highs + closes) + closes) + closes

        part_d = cs_rank(closes) + (part_d1 + part_d2 + part_d3) + highs + closes - (opens - lows)

        denominator_inner = rank_closes + amounts + (cs_rank(closes) + part_d)
        denominator = cs_rank(cs_rank(denominator_inner))

        self.factor_data = numerator / denominator.where(denominator != 0, np.nan)


class gp_alpha_026(factor):
    def set_factor_property(self):
        """
        GP Alpha#26: With division by -1
        Formula: Contains div(-1, closes) term

        Performance Metrics (ID: 11):
        - Fitness: 1377.5275, IR: 2.6952, Net IR: -0.0000, t-stat: 2.5494, Max DD: 1303.0078
        """
        factor_property = {
            'factor_name': 'gp_alpha_026',
            'data_needed': ['close', 'open', 'high', 'low', 'amount'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        amounts = self.data['amount']

        # Numerator with div(-1, closes)
        inner1 = highs + (closes + closes + closes)
        rank_inner1 = cs_rank(inner1)

        # div(-1, closes) term
        div_term = -1 / closes.where(closes != 0, np.nan)

        part_a1 = amounts + (closes + highs) + closes
        part_a2 = amounts + part_a1 + highs + closes
        part_a3 = div_term + cs_rank(closes) + amounts + (amounts + (highs + closes) + closes) + closes
        part_a = part_a2 + part_a3 + closes

        numerator_inner = rank_inner1 + amounts + part_a
        numerator = cs_rank(numerator_inner)

        # Denominator
        rank_closes = cs_rank(closes)
        part_b1 = lows + highs + closes
        part_b2 = ts_mean(amounts, 5)
        part_b3 = amounts + (amounts + (highs + closes) + closes) + closes

        part_b = cs_rank(closes) + (part_b1 + part_b2 + part_b3) + highs + closes - (opens - lows)

        denominator_inner = rank_closes + amounts + (cs_rank(closes) + part_b)
        denominator = cs_rank(cs_rank(denominator_inner))

        self.factor_data = numerator / denominator.where(denominator != 0, np.nan)


class gp_alpha_027(factor):
    def set_factor_property(self):
        """
        GP Alpha#27: Minimal highs structure
        Formula: Simplified with just highs in numerator

        Performance Metrics (ID: 35, 39, 43, 44, 47):
        - Fitness: 1377.5275, IR: 2.6952, Net IR: -0.0000, t-stat: 2.5494, Max DD: 1303.0078
        """
        factor_property = {
            'factor_name': 'gp_alpha_027',
            'data_needed': ['close', 'high', 'low', 'amount'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        amounts = self.data['amount']

        # Numerator: very simple
        inner1 = highs + (closes + closes + closes)
        rank_inner1 = cs_rank(inner1)
        numerator_inner = rank_inner1 + highs
        numerator = cs_rank(numerator_inner)

        # Denominator
        rank_closes = cs_rank(closes)
        part_b1 = lows + highs + amounts
        part_b2 = ts_mean(amounts, 5)
        part_b3 = amounts + (amounts + (highs + closes) + amounts) + closes

        part_b = cs_rank(closes) + (part_b1 + part_b2 + part_b3) + highs + closes - (opens - lows)

        denominator_inner = rank_closes + amounts + (cs_rank(closes) + part_b)
        denominator = cs_rank(cs_rank(denominator_inner))

        self.factor_data = numerator / denominator.where(denominator != 0, np.nan)


class gp_alpha_028(factor):
    def set_factor_property(self):
        """
        GP Alpha#28: With sub(8, 9) term
        Formula: Contains sub(8, 9) = -1 term

        Performance Metrics (ID: 19, 32):
        - Fitness: 1377.5275, IR: 2.6952, Net IR: -0.0000, t-stat: 2.5494, Max DD: 1303.0078
        """
        factor_property = {
            'factor_name': 'gp_alpha_028',
            'data_needed': ['close', 'open', 'high', 'low', 'amount'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        amounts = self.data['amount']

        # Numerator
        inner1 = highs + (closes + closes + closes)
        rank_inner1 = cs_rank(inner1)

        part_a = amounts + cs_rank(closes) + amounts + amounts
        part_b = amounts + (closes + highs) + closes + highs + closes
        part_c = amounts + part_a + closes

        numerator_inner = rank_inner1 + amounts + part_b + part_c
        numerator = cs_rank(numerator_inner)

        # Denominator with sub(8, 9) = -1
        rank_closes = cs_rank(closes)
        part_d1 = lows + highs + closes
        part_d2 = ts_mean(amounts, 5)
        part_d3 = amounts + (amounts + (highs + closes) + (8 - 9)) + closes  # sub(8, 9) = -1

        part_d = cs_rank(closes) + (part_d1 + part_d2 + part_d3) + highs + closes - (opens - lows)

        denominator_inner = rank_closes + amounts + (cs_rank(closes) + part_d)
        denominator = cs_rank(cs_rank(denominator_inner))

        self.factor_data = numerator / denominator.where(denominator != 0, np.nan)


class gp_alpha_029(factor):
    def set_factor_property(self):
        """
        GP Alpha#29: With div(closes, lows) term (ID: 20)
        """
        factor_property = {
            'factor_name': 'gp_alpha_029',
            'data_needed': ['close', 'open', 'high', 'low', 'amount'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        amounts = self.data['amount']

        inner1 = highs + (closes + closes + closes)
        rank_inner1 = cs_rank(inner1)

        div_cl = closes / lows.where(lows != 0, np.nan)
        part_a = amounts + cs_rank(closes) + (highs + closes) + div_cl
        part_b = amounts + (closes + highs) + closes + highs + closes
        part_c = amounts + part_a + closes

        numerator = cs_rank(rank_inner1 + amounts + part_b + part_c)

        rank_closes = cs_rank(closes)
        part_d = cs_rank(closes) + (lows + highs + closes + ts_mean(amounts, 5) + amounts + (amounts + (highs + closes) + closes) + closes) + highs + closes - (opens - lows)
        denominator = cs_rank(cs_rank(rank_closes + amounts + (cs_rank(closes) + part_d)))

        self.factor_data = numerator / denominator.where(denominator != 0, np.nan)


class gp_alpha_030(factor):
    def set_factor_property(self):
        """
        GP Alpha#30: High-fitness complex factor
        Performance Metrics: Fitness: 1377.5275, IR: 2.6952
        """
        factor_property = {
            'factor_name': 'gp_alpha_030',
            'data_needed': ['close', 'open', 'high', 'low', 'amount'],
            'factor_type': 'market'
        }
        return factor_property

    def calc_factor(self):
        closes = self.data['close']
        opens = self.data['open']
        highs = self.data['high']
        lows = self.data['low']
        amounts = self.data['amount']

        # Numerator: rank-based complex structure
        inner1 = highs + (closes + closes + closes)
        rank_inner1 = cs_rank(inner1)

        # Numerator components (simplified general form)
        part_a = amounts + (closes + highs) + closes
        part_b = amounts + part_a + highs + closes
        part_c = amounts + cs_rank(closes) + amounts + closes

        numerator_inner = rank_inner1 + amounts + part_b + part_c
        numerator = cs_rank(numerator_inner)

        # Denominator: double-rank nested structure
        rank_closes = cs_rank(closes)
        part_d1 = lows + highs + closes
        part_d2 = ts_mean(amounts, 5)
        part_d3 = amounts + (amounts + (highs + closes) + closes) + closes

        part_d = cs_rank(closes) + (part_d1 + part_d2 + part_d3) + highs + closes - (opens - lows)
        denominator_inner = rank_closes + amounts + (cs_rank(closes) + part_d)
        denominator = cs_rank(cs_rank(denominator_inner))

        # Final factor
        self.factor_data = numerator / denominator.where(denominator != 0, np.nan)
