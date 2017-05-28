from .embed_eed_eeq import EmbedEEdEEq
from .embed_eeq_eeq import EmbedEEqEEq
from .embed_eed_me import EmbedEEdME
from .embed_eeq_me import EmbedEEqME
from .embed_eed_none import EmbedEEdNone
from .embed_eeq_none import EmbedEEqNone
from .embed_none_none import EmbedNoneNone

__all__ = ['choose_embedtool']

EMBEDTOOLS = ['EmbedEEdEEq', 'EmbedEEqEEq', 'EmbedEEdME', 'EmbedEEqME',
              'EmbedEEdNone', 'EmbedEEqNone', 'EmbedNoneNone']

def choose_embedtool(qmEmbedNear, qmEmbedFar):
    if qmEmbedNear is None:
        qmEmbedNear = 'None'

    if qmEmbedFar is None:
        qmEmbedFar = 'None'

    for embedtool in EMBEDTOOLS:
        embedtool = globals()[embedtool]
        if embedtool.check_embed(qmEmbedNear, qmEmbedFar):
            return embedtool
    raise ValueError("Cannot use '{}' for far field while using '{}' for near field.".format(qmEmbedNear, qmEmbedFar))
