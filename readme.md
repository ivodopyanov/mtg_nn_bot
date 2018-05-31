# Magic: the Gathering draft bot based on neural network (Tensorflow)
It uses card embeddings + attention

**mtg_nn_bot/settings.json** - settings for model training
**mtg_nn_bot/model.py** - Tensorflow model itself
**mtg_nn_bot/model.py** - code for training model from previously recorded draft dataset (if such is accessible)
**mtg_nn_bot/service/draft_controller.py** - wrapper for model, which modifies pick records for draft
**mtg_nn_bot/service/console_service.py** - application to play draft from console using model
**mtg_nn_bot/tests/test_model.py** - runs model on some drafts from dataset and generates text output to compare AI picks with human picks
**mtg_nn_bot/tests/test_draft_controller.py** - runs draft prediction for random drafts through draft controller and generates text output to validate it by hand