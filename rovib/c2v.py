# C2v symmetry labels of Wang's rotational functions, keys = (k%2, tau)
C2V_KTAU_IRREPS = {(0, 0): "A1", (0, 1): "B1", (1, 0): "B2", (1, 1): "A2"}

C2V_IRREPS = ("A1", "A2", "B1", "B2")

C2V_PRODUCT_TABLE = {
    ("A1", "A1"): "A1",
    ("A1", "A2"): "A2",
    ("A1", "B1"): "B1",
    ("A1", "B2"): "B2",
    #
    ("A2", "A1"): "A2",
    ("A2", "A2"): "A1",
    ("A2", "B1"): "B2",
    ("A2", "B2"): "B1",
    #
    ("B1", "A1"): "B1",
    ("B1", "A2"): "B2",
    ("B1", "B1"): "A1",
    ("B1", "B2"): "A2",
    #
    ("B2", "A1"): "B2",
    ("B2", "A2"): "B1",
    ("B2", "B1"): "A2",
    ("B2", "B2"): "A1",
}