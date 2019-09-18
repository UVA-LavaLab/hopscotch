###############################################################################
#
# Makefile helper. Forward $(MAKE) calls to the subdirectories specified by
# $(SUBDIRS). This file is included by the top level makefiles.
#
# Author: 	Alif Ahmed
# email: 	alifahmed@virginia.edu
# Updated: 	Aug 06, 2019
#
###############################################################################

CLEAN_DIRS = $(addsuffix .clean,$(SUBDIRS))

.PHONY: all clean $(SUBDIRS) $(CLEAN_DIRS)

all: $(SUBDIRS)

clean: $(CLEAN_DIRS)

$(SUBDIRS):
	@$(MAKE) -C $@ --no-print-directory

$(CLEAN_DIRS): %.clean :
	@$(MAKE) -C $* --no-print-directory clean


