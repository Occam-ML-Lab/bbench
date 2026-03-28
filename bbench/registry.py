"""Dataset registry: names, metadata, and category helpers."""

from dataclasses import dataclass

def hf_repo_id() -> str:
    return "OccaMLab/bayesian-benchmarks"


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    size: int
    dim: int
    category: str
    subcategory: str
    num_classes: int | None = None
    md5: str | None = None


# fmt: off
def dataset_registry() -> list[DatasetMeta]:
  return [
    # UCI regression
    DatasetMeta("boston",      506,     13, "regression", "uci", md5="eb6957a328e026f9126e49c450febec8"),
    DatasetMeta("concrete",   1030,     8, "regression", "uci", md5="2fadcfc3bc73779676f7f9e5cc03f831"),
    DatasetMeta("energy",      768,     8, "regression", "uci", md5="65273da7e1f05dd4c96ef49bed03167e"),
    DatasetMeta("kin8nm",     8192,     8, "regression", "uci", md5="62ca34f8db7f4a7752c03547863d6d15"),
    DatasetMeta("naval",     11934,    14, "regression", "uci", md5="8cb5f5adb54a2d44c92feeb1ffb59fb2"),
    DatasetMeta("power",      9568,     4, "regression", "uci", md5="023f4e4fc0d63bba69c2aa44f7245419"),
    DatasetMeta("protein",   45730,     9, "regression", "uci", md5="c40979769aad006e9b28f1f26b3d2436"),
    DatasetMeta("winered",    1599,    11, "regression", "uci", md5="c9155915c6b36ce7c023fce6d221e9d3"),
    DatasetMeta("winewhite",  4898,    11, "regression", "uci", md5="fd7cd466d154b16a597703090764f43d"),
    DatasetMeta("yacht",       308,     6, "regression", "uci", md5="711be10bd1cd1b71dd012e66760555ee"),
    # Wilson regression
    DatasetMeta("3droad",        434874,  3, "regression", "wilson", md5="cc026800ebafb364c4327642407694b1"),
    DatasetMeta("challenger",        23,  4, "regression", "wilson", md5="e9d6f97297083f0a1f1b63eaaf84c8f6"),
    DatasetMeta("gas",             2565,128, "regression", "wilson", md5="0b4578cc581afeeb8e979020c2db2676"),
    DatasetMeta("servo",            167,  4, "regression", "wilson", md5="0c6226976345f5a290c9b1bc5eeb5dac"),
    DatasetMeta("tamielectric",   45781,  3, "regression", "wilson", md5="2c85c205c9c707551f286165e124f39a"),
    DatasetMeta("airfoil",         1503,  5, "regression", "wilson", md5="0c567a006fc02ec25b2300e1a19e40b2"),
    DatasetMeta("machine",          209,  7, "regression", "wilson", md5="70df13756783b4155f4609064ecea1d7"),
    DatasetMeta("skillcraft",      3338, 19, "regression", "wilson", md5="fef9525dccc1f51ad302d82fe451fa70"),
    DatasetMeta("autompg",          392,  7, "regression", "wilson", md5="71fde045a85463fb50238ddf8277a83d"),
    DatasetMeta("concreteslump",    103,  7, "regression", "wilson", md5="05a2b8f5cf24a759399a95e6694ea4f8"),
    DatasetMeta("houseelectric", 2049280,11, "regression", "wilson", md5="21694cd4ce4b76f7ff6a9e33cb55e7fd"),
    DatasetMeta("parkinsons",      5875, 20, "regression", "wilson", md5="68ed3c13cc5804710b2998941234d43f"),
    DatasetMeta("slice",          53500,385, "regression", "wilson", md5="f6138817a1a09bd59f69421222f87792"),
    DatasetMeta("autos",            159, 25, "regression", "wilson", md5="aa7f97c353d2625099d2cdefcbe55096"),
    DatasetMeta("elevators",      16599, 18, "regression", "wilson", md5="d749f4aa525584ab5a45eb350ed2e84c"),
    DatasetMeta("pendulum",         630,  9, "regression", "wilson", md5="fa43e92a7e77e4ede430ad21c4897787"),
    DatasetMeta("sml",             4137, 26, "regression", "wilson", md5="b151f5627cc22edd0a0c291028500159"),
    DatasetMeta("bike",           17379, 17, "regression", "wilson", md5="1f18f32957c38d7707ca0134385a27c7"),
    DatasetMeta("keggdirected",   48827, 20, "regression", "wilson", md5="dc1419f59c9728cc8c1b5ab1d2866d79"),
    DatasetMeta("pol",            15000, 26, "regression", "wilson", md5="7f4da6ec553c86df108afc193b24d9b7"),
    DatasetMeta("solar",           1066, 10, "regression", "wilson", md5="6bbb27a894aecd2c6521ed174f6572e4"),
    DatasetMeta("breastcancer",     194, 33, "regression", "wilson", md5="6ca4c9a73fef5912f5b5edff0dc3f588"),
    DatasetMeta("fertility",        100,  9, "regression", "wilson", md5="d292fca4c0efc55cafc2c681c4111c2a"),
    DatasetMeta("keggundirected", 63608, 27, "regression", "wilson", md5="f43460f6a5abd2d7ce75f82cf4ef0c9c"),
    DatasetMeta("song",          515345, 90, "regression", "wilson", md5="cc2e73b8f5487d78913e8e71d5c2e8bb"),
    DatasetMeta("buzz",          583250, 77, "regression", "wilson", md5="afa11bdaea568ec6653424fd9a7e5b54"),
    DatasetMeta("forest",           517, 12, "regression", "wilson", md5="ad9f73852ece0c8b3de2892ffd568213"),
    DatasetMeta("kin40k",         40000,  8, "regression", "wilson", md5="109e7eecae2b7ec6ea022004435ea7de"),
    DatasetMeta("pumadyn32nm",     8192, 32, "regression", "wilson", md5="dd874ff825a8a3f1594dc97edf016868"),
    DatasetMeta("stock",            536, 11, "regression", "wilson", md5="d42bfb288118340b7e16739cdcd87038"),
    # NY Taxi regression
    DatasetMeta("NYTaxiTimePrediction", 1420068, 8, "regression", "taxi"),
    # Mujoco RL dynamics
    DatasetMeta("Ant-v2",                      0, 119, "reinforcement", "mujoco"),
    DatasetMeta("HalfCheetah-v2",              0,  23, "reinforcement", "mujoco"),
    DatasetMeta("Hopper-v2",                   0,  14, "reinforcement", "mujoco"),
    DatasetMeta("Humanoid-v2",                 0, 393, "reinforcement", "mujoco"),
    DatasetMeta("InvertedDoublePendulum-v2",   0,  12, "reinforcement", "mujoco"),
    DatasetMeta("InvertedPendulum-v2",         0,   5, "reinforcement", "mujoco"),
    DatasetMeta("Pendulum-v0",                 0,   4, "reinforcement", "mujoco"),
    DatasetMeta("Reacher-v2",                  0,  13, "reinforcement", "mujoco"),
    DatasetMeta("Swimmer-v2",                  0,  10, "reinforcement", "mujoco"),
    DatasetMeta("Walker2d-v2",                 0,  23, "reinforcement", "mujoco"),
    # Classification
    DatasetMeta("heart-va",                        200,  13, "classification", "classification", num_classes=5, md5="bf9f9814a55991ef9b42025dc901f1e8"),
    DatasetMeta("connect-4",                     67557,  43, "classification", "classification", num_classes=2, md5="30e540a046ea067a6859d9e153e46347"),
    DatasetMeta("wine",                            178,  14, "classification", "classification", num_classes=3, md5="d013d4a6b18e2923081da9f40cf21bf1"),
    DatasetMeta("tic-tac-toe",                     958,  10, "classification", "classification", num_classes=2, md5="9582b162457bc7736175ac6604cccb6e"),
    DatasetMeta("fertility_diagnosis",             100,  10, "classification", "classification", num_classes=2, md5="125b9d15e465d5fc4c27e0c7ecdb2c26"),
    DatasetMeta("statlog-german-credit",          1000,  25, "classification", "classification", num_classes=2, md5="998b81ba4347db53417c969fe77060cd"),
    DatasetMeta("car",                            1728,   7, "classification", "classification", num_classes=4, md5="848863b3c41f2cdbb9c5cde608e8da67"),
    DatasetMeta("libras",                          360,  91, "classification", "classification", num_classes=15, md5="d04d54cb617440dcb5139fae9299b402"),
    DatasetMeta("spambase",                       4601,  58, "classification", "classification", num_classes=2, md5="19eec8c3d53db5975ae6ffed36a4d3b1"),
    DatasetMeta("pittsburg-bridges-MATERIAL",      106,   8, "classification", "classification", num_classes=3, md5="c9fd59ed3ffbdafb81a61e9f03612834"),
    DatasetMeta("hepatitis",                       155,  20, "classification", "classification", num_classes=2, md5="ebcf6783e6e591918abf55a572ef97a2"),
    DatasetMeta("acute-inflammation",              120,   7, "classification", "classification", num_classes=2, md5="f23c34f2ea8d42a317b6776d377a50e3"),
    DatasetMeta("pittsburg-bridges-TYPE",          105,   8, "classification", "classification", num_classes=6, md5="901109fba83fcf378d8b404ac6a6f75f"),
    DatasetMeta("arrhythmia",                      452, 263, "classification", "classification", num_classes=13, md5="971aa1b0d808a994712b1646884b9db8"),
    DatasetMeta("musk-2",                         6598, 167, "classification", "classification", num_classes=2, md5="7ca5e65ea02f8385c3754213e61cd485"),
    DatasetMeta("twonorm",                        7400,  21, "classification", "classification", num_classes=2, md5="0c0ffa2f5faa6818ed85d831454aa4e2"),
    DatasetMeta("nursery",                       12960,   9, "classification", "classification", num_classes=5, md5="56c3d2ba9693bf52920c1d61d536e369"),
    DatasetMeta("breast-cancer-wisc-prog",         198,  34, "classification", "classification", num_classes=2, md5="36f29fa52ed771d135be617db3a5c587"),
    DatasetMeta("seeds",                           210,   8, "classification", "classification", num_classes=3, md5="fe93f37e4f16a50b9117c3e6c4ac7575"),
    DatasetMeta("lung-cancer",                      32,  57, "classification", "classification", num_classes=3, md5="145e724ab86fef6260519f2ed9a172be"),
    DatasetMeta("waveform",                       5000,  22, "classification", "classification", num_classes=3, md5="f15012aafe0ff99994848e4c39118dd8"),
    DatasetMeta("audiology-std",                   196,  60, "classification", "classification", num_classes=18, md5="ca73f73144b2025bd74ab767c8a7ad06"),
    DatasetMeta("trains",                           10,  30, "classification", "classification", num_classes=2, md5="41b99726dcc5406b22d5f9e65a6c0ac1"),
    DatasetMeta("horse-colic",                     368,  26, "classification", "classification", num_classes=2, md5="7e58dcb3b80ab7e87c749340021d79a8"),
    DatasetMeta("miniboone",                    130064,  51, "classification", "classification", num_classes=2, md5="a33a842313e8ca736089fa71007eba3a"),
    DatasetMeta("pittsburg-bridges-SPAN",           92,   8, "classification", "classification", num_classes=3, md5="29f39b570d41c3e8a8c59244536fa9b2"),
    DatasetMeta("breast-cancer-wisc-diag",         569,  31, "classification", "classification", num_classes=2, md5="2555279e43ce5599381706821627ceab"),
    DatasetMeta("statlog-heart",                   270,  14, "classification", "classification", num_classes=2, md5="76b3d372fc0ffcb006b616e2406dcf13"),
    DatasetMeta("blood",                           748,   5, "classification", "classification", num_classes=2, md5="6bc94661a6fcc059b80e9faf127fa365"),
    DatasetMeta("primary-tumor",                   330,  18, "classification", "classification", num_classes=15, md5="03fa904710665078438318bc1eddde18"),
    DatasetMeta("cylinder-bands",                  512,  36, "classification", "classification", num_classes=2, md5="3ab869de5f2735fedbda474ddf52d402"),
    DatasetMeta("glass",                           214,  10, "classification", "classification", num_classes=6, md5="9c4c3161fbc9f3a2c651b733313421a2"),
    DatasetMeta("contrac",                        1473,  10, "classification", "classification", num_classes=3, md5="8e7f73a2d3029a6dc5720031472f2e07"),
    DatasetMeta("statlog-shuttle",               58000,  10, "classification", "classification", num_classes=7, md5="00ff29d02a8623baea2f6a3c9dfac1ff"),
    DatasetMeta("zoo",                             101,  17, "classification", "classification", num_classes=7, md5="87fed9ddeca670a8d1cf4169941582a5"),
    DatasetMeta("musk-1",                          476, 167, "classification", "classification", num_classes=2, md5="22cab6cc9d662e52fe57abfd439285c7"),
    DatasetMeta("hill-valley",                    1212, 101, "classification", "classification", num_classes=2, md5="c33c60b9204006307c5c7f5386ca64a0"),
    DatasetMeta("hayes-roth",                      160,   4, "classification", "classification", num_classes=3, md5="8d804545827a82acec05d611bcddc648"),
    DatasetMeta("optical",                        5620,  63, "classification", "classification", num_classes=10, md5="accb4046628c6a24b3e1c4da7e3d9850"),
    DatasetMeta("credit-approval",                 690,  16, "classification", "classification", num_classes=2, md5="0bf0e1d3b1913d18b8a7e25bb254f397"),
    DatasetMeta("pendigits",                     10992,  17, "classification", "classification", num_classes=10, md5="859a2d612e8896e23991a9570aca61ae"),
    DatasetMeta("pittsburg-bridges-REL-L",         103,   8, "classification", "classification", num_classes=3, md5="50a90b4bea8aef823abd4f14eebf26f5"),
    DatasetMeta("dermatology",                     366,  35, "classification", "classification", num_classes=6, md5="1f0ed1e13067ee27d3818b550679c8f1"),
    DatasetMeta("soybean",                         683,  36, "classification", "classification", num_classes=18, md5="4c5eaf3c4f8e2d3b64466609ca9656cf"),
    DatasetMeta("ionosphere",                      351,  34, "classification", "classification", num_classes=2, md5="00cc53345d3da4fdf25dc425c72a5b31"),
    DatasetMeta("planning",                        182,  13, "classification", "classification", num_classes=2, md5="b86d96a3d5243712efd14b344df0b9f4"),
    DatasetMeta("energy-y1",                       768,   9, "classification", "classification", num_classes=3, md5="65cb307632d01c0f1331d34dbf24620f"),
    DatasetMeta("acute-nephritis",                 120,   7, "classification", "classification", num_classes=2, md5="6ab375378095ce8341bc33474e6debda"),
    DatasetMeta("pittsburg-bridges-T-OR-D",        102,   8, "classification", "classification", num_classes=2, md5="82c2ad8fc3697f3ac271d63db4c69be5"),
    DatasetMeta("letter",                        20000,  17, "classification", "classification", num_classes=26, md5="29a51a6e2200390eac7b8f5aee5d654c"),
    DatasetMeta("titanic",                        2201,   4, "classification", "classification", num_classes=2, md5="0cdac2f97ea0f5c075c3626e65cc027c"),
    DatasetMeta("adult",                         48842,  15, "classification", "classification", num_classes=2, md5="0179276da07a6543afbaa8ee286244be"),
    DatasetMeta("lymphography",                    148,  19, "classification", "classification", num_classes=4, md5="3f1b429a2c24ac2912cbb5c5ae608c91"),
    DatasetMeta("statlog-australian-credit",       690,  15, "classification", "classification", num_classes=2, md5="b6948b33f6ed2da8840aac568eac7139"),
    DatasetMeta("chess-krvk",                    28056,   7, "classification", "classification", num_classes=18, md5="7f8728fd4f7b94cd49608bb21dd782ca"),
    DatasetMeta("bank",                           4521,  17, "classification", "classification", num_classes=2, md5="dc930d02ed443612dca5596e2033aa84"),
    DatasetMeta("statlog-landsat",                6435,  37, "classification", "classification", num_classes=6, md5="effaedd39cb4c74e61bfeb2985f2609c"),
    DatasetMeta("heart-hungarian",                 294,  13, "classification", "classification", num_classes=2, md5="76e8d89e879943221f8ab4440a7f244f"),
    DatasetMeta("flags",                           194,  29, "classification", "classification", num_classes=8, md5="8206f39b2a09f0ec68fdb235d3fd9230"),
    DatasetMeta("mushroom",                       8124,  22, "classification", "classification", num_classes=2, md5="5dae7338f3de5df91f670e846ded94cb"),
    DatasetMeta("conn-bench-sonar-mines-rocks",    208,  61, "classification", "classification", num_classes=2, md5="7345cc51bee345fcfaf2d0366f90a9a8"),
    DatasetMeta("image-segmentation",             2310,  19, "classification", "classification", num_classes=7, md5="a65105caed5e18e55bdbd143a10f111a"),
    DatasetMeta("congressional-voting",            435,  17, "classification", "classification", num_classes=2, md5="18a90a49aa7e993d99570f9e3ae7c248"),
    DatasetMeta("annealing",                       898,  32, "classification", "classification", num_classes=5, md5="be6d5319d78f5dd5882df6e4efbbe3d1"),
    DatasetMeta("semeion",                        1593, 257, "classification", "classification", num_classes=10, md5="86ad48a90152a246b92cb33f2dcd6d2e"),
    DatasetMeta("echocardiogram",                  131,  11, "classification", "classification", num_classes=2, md5="ce7809094c87760f3b2a38b14acf7f0f"),
    DatasetMeta("statlog-image",                  2310,  19, "classification", "classification", num_classes=7, md5="f05c981b13355cbbc36612d3b31cc2e4"),
    DatasetMeta("wine-quality-white",             4898,  12, "classification", "classification", num_classes=7, md5="c000a51290abcc01aae059eaaa91e407"),
    DatasetMeta("lenses",                           24,   5, "classification", "classification", num_classes=3, md5="0747756b2210b9469f44c2ec2d924a78"),
    DatasetMeta("plant-margin",                   1600,  65, "classification", "classification", num_classes=100, md5="f4ee122364e56272db1f300c1c015cbf"),
    DatasetMeta("post-operative",                   90,   9, "classification", "classification", num_classes=3, md5="c7dfaf24fbc55ac338f16c580a4cf44d"),
    DatasetMeta("thyroid",                        7200,  22, "classification", "classification", num_classes=3, md5="f4c18c326ef55daa06170e7193489f4f"),
    DatasetMeta("monks-2",                         601,   7, "classification", "classification", num_classes=2, md5="8c7909dd30f48fabed3ddf5a37cbae3f"),
    DatasetMeta("molec-biol-promoter",             106,  58, "classification", "classification", num_classes=2, md5="bec7745a5353dd1d5a81e4a054b2eb19"),
    DatasetMeta("chess-krvkp",                    3196,  37, "classification", "classification", num_classes=2, md5="084bcbac029e8ca1fbd76d366480b414"),
    DatasetMeta("balloons",                         16,   5, "classification", "classification", num_classes=2, md5="3f1560b46e9bd22178c525a457a8b56d"),
    DatasetMeta("low-res-spect",                   531, 101, "classification", "classification", num_classes=9, md5="2f7ed7fb3f3a03d4c12487a76b1f8e1d"),
    DatasetMeta("plant-texture",                  1599,  65, "classification", "classification", num_classes=100, md5="d2700eeefb3c8fee1c18044818f71767"),
    DatasetMeta("haberman-survival",               306,   4, "classification", "classification", num_classes=2, md5="5b421e54ca9401a1f3e83bd06e754dc9"),
    DatasetMeta("spect",                           265,  23, "classification", "classification", num_classes=2, md5="b5b65d02fb8ffb6856f7849c2c6c68d3"),
    DatasetMeta("plant-shape",                    1600,  65, "classification", "classification", num_classes=100, md5="39794aa810ed1ce6545470e98208eac2"),
    DatasetMeta("parkinsons_diagnosis",            195,  23, "classification", "classification", num_classes=2, md5="7e46ead92fef16df43ec1de90cf2f2a4"),
    DatasetMeta("oocytes_merluccius_nucleus_4d",  1022,  42, "classification", "classification", num_classes=2, md5="41b1e86c0de79d33adf218163e32830c"),
    DatasetMeta("conn-bench-vowel-deterding",      990,  12, "classification", "classification", num_classes=11, md5="acdd60166973a511c5b57d003176bd7b"),
    DatasetMeta("ilpd-indian-liver",               583,  10, "classification", "classification", num_classes=2, md5="18832c8dd477941e44c86a983ced4287"),
    DatasetMeta("heart-cleveland",                 303,  14, "classification", "classification", num_classes=5, md5="49bbcf8530582697b2e8250703fb30de"),
    DatasetMeta("synthetic-control",               600,  61, "classification", "classification", num_classes=6, md5="0c39121b6d42d1b766214f020ecd10a6"),
    DatasetMeta("vertebral-column-2clases",        310,   7, "classification", "classification", num_classes=2, md5="deb8f38b86d4759a14228b98db6fadd0"),
    DatasetMeta("teaching",                        151,   6, "classification", "classification", num_classes=3, md5="2e4be7a8cf98fa89bb5e902cdbcf1313"),
    DatasetMeta("cardiotocography-10clases",      2126,  22, "classification", "classification", num_classes=10, md5="6dca837a27b44d6452b6f66c9e0d0438"),
    DatasetMeta("heart-switzerland",               123,  13, "classification", "classification", num_classes=5, md5="b9e674f104248ab50945f4a0913baaca"),
    DatasetMeta("led-display",                    1000,   8, "classification", "classification", num_classes=10, md5="91e8991a6e4bb7aed660b6edcd36c91b"),
    DatasetMeta("molec-biol-splice",              3190,  61, "classification", "classification", num_classes=3, md5="7e5781cf98eb48ad84deab2b484a4da6"),
    DatasetMeta("wall-following",                 5456,  25, "classification", "classification", num_classes=4, md5="64062e5d45820d2305c3f94feb5dfa3a"),
    DatasetMeta("statlog-vehicle",                 846,  19, "classification", "classification", num_classes=4, md5="23eea31b235f47778864a60289df2460"),
    DatasetMeta("ringnorm",                       7400,  21, "classification", "classification", num_classes=2, md5="2b4d3fa77161d5214d1ac9aeec92804b"),
    DatasetMeta("energy-y2",                       768,   9, "classification", "classification", num_classes=3, md5="7d417dd54f699e86c52f103ae60ebef9"),
    DatasetMeta("oocytes_trisopterus_nucleus_2f",  912,  26, "classification", "classification", num_classes=2, md5="2edcf30bd8fd65068956d6707c54c508"),
    DatasetMeta("yeast",                          1484,   9, "classification", "classification", num_classes=10, md5="c2ea981a14bd7d9bda1ec129e8a6c6dc"),
    DatasetMeta("oocytes_merluccius_states_2f",   1022,  26, "classification", "classification", num_classes=3, md5="f9b8886725a33d6d45535070ac6f610f"),
    DatasetMeta("oocytes_trisopterus_states_5b",   912,  33, "classification", "classification", num_classes=3, md5="ef99c6e79c1cfc1b0c82cbea491535f6"),
    DatasetMeta("breast-cancer-wisc",              699,  10, "classification", "classification", num_classes=2, md5="7dde0e57f669c9e8749c235fe4ac9a67"),
    DatasetMeta("steel-plates",                   1941,  28, "classification", "classification", num_classes=7, md5="cb0bbbf0877472a3b48adc300eeaf45b"),
    DatasetMeta("mammographic",                    961,   6, "classification", "classification", num_classes=2, md5="aa66c4b7cdf1a2484604237c73ef500f"),
    DatasetMeta("monks-3",                         554,   7, "classification", "classification", num_classes=2, md5="ffa93f940fa792e74bffaa96c09537df"),
    DatasetMeta("balance-scale",                   625,   5, "classification", "classification", num_classes=3, md5="d824e9f1a5e06c0eee7b8bdec56ef950"),
    DatasetMeta("ecoli",                           336,   8, "classification", "classification", num_classes=8, md5="45849b02ca814f6be96eec090e388927"),
    DatasetMeta("spectf",                          267,  45, "classification", "classification", num_classes=2, md5="c4aa677bb9c509096a2b9f532c63367d"),
    DatasetMeta("monks-1",                         556,   7, "classification", "classification", num_classes=2, md5="ac6f1be9b22382ebd5619e0cc7ed305c"),
    DatasetMeta("page-blocks",                    5473,  11, "classification", "classification", num_classes=5, md5="d74bf845501af5aa3e1d7ff813ed32b8"),
    DatasetMeta("magic",                         19020,  11, "classification", "classification", num_classes=2, md5="c5b1c55dd18152776bf4e44df205e45c"),
    DatasetMeta("pima",                            768,   9, "classification", "classification", num_classes=2, md5="c0966d3818b475968b301bf765115444"),
    DatasetMeta("breast-tissue",                   106,  10, "classification", "classification", num_classes=6, md5="cad1ef06283f5da95bd77de417137ac0"),
    DatasetMeta("ozone",                          2536,  73, "classification", "classification", num_classes=2, md5="c35526f97211e89fd1bcc7b034c151e7"),
    DatasetMeta("iris",                            150,   5, "classification", "classification", num_classes=3, md5="e383a18bf3358204d1fc77522661f01c"),
    DatasetMeta("waveform-noise",                 5000,  41, "classification", "classification", num_classes=3, md5="d32195362dc8e8ca5bf7feb98b2a8615"),
    DatasetMeta("cardiotocography-3clases",       2126,  22, "classification", "classification", num_classes=3, md5="d4f77767541d58d80893f1d41fe491a2"),
    DatasetMeta("wine-quality-red",               1599,  12, "classification", "classification", num_classes=6, md5="6b0142245232d327dd9932d23cac6f7c"),
    DatasetMeta("vertebral-column-3clases",        310,   7, "classification", "classification", num_classes=3, md5="3a4951d95846517ca85cb7159ee432c9"),
    DatasetMeta("breast-cancer",                   286,  10, "classification", "classification", num_classes=2, md5="76efc9ed2bc5496fb625a1861ff5dcbd"),
    DatasetMeta("abalone",                        4177,   9, "classification", "classification", num_classes=3, md5="649e9b2c0df7914f09299231b32e3ee1"),
  ]
# fmt: on


def __meta_by_name() -> dict[str, DatasetMeta]:
    return {d.name: d for d in dataset_registry()}


def regression_datasets() -> list[str]:
    return sorted(d.name for d in dataset_registry() if d.category == "regression")


def classification_datasets() -> list[str]:
    return sorted(d.name for d in dataset_registry() if d.category == "classification")


def reinforcement_datasets() -> list[str]:
    return sorted(d.name for d in dataset_registry() if d.category == "reinforcement")


def all_datasets() -> list[str]:
    return sorted(__meta_by_name())


def dataset_info(name: str) -> DatasetMeta:
    return __meta_by_name()[name]
