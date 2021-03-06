"""Reaction graph for Figure 5C from the PySB publication"""

import pysb.tools.render_reactions

from earm.mito.lopez_embedded import model

# print out the graphviz representation of the contact map
print pysb.tools.render_reactions.run(model)

