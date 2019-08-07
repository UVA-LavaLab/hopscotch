###############################################################################
#
# Makefile helper. Forward $(MAKE) calls to the subdirectories specified by
# $(TARGETS). This file is included by the top level makefiles.
#
# Author: 	Alif Ahmed
# email: 	alifahmed@virginia.edu
# Updated: 	Aug 06, 2019
#
###############################################################################

CLEAN_TARGETS = $(addsuffix .clean,$(TARGETS))

all: $(TARGETS)

$(TARGETS):
	@$(MAKE) -C $@ --no-print-directory

clean: $(CLEAN_TARGETS)
	@rm -rf *.so

%.clean: %
	@$(MAKE) clean -C $< --no-print-directory

.PHONY: all clean $(TARGETS)
