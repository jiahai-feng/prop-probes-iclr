import os.path as osp
from dotenv import load_dotenv

COREF_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))

load_dotenv(osp.join(COREF_ROOT, '.env'), verbose=True)
import site

site.addsitedir(osp.join(COREF_ROOT, "TransformerLens"))
