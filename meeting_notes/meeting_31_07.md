# Twelth Meeting - 31/07

## Outline

- Discussed work done in AIML425, regarding building regression deep neural network models and the pipeline involved.
- Discussed the tests used when designing the classes and the model tuning techniques involved, which can later be used for evalutation for PrivML model.
- Discussed current ML pipeline that has been implemented and how it will use existing work done.
- Discussed whether the UI should be application-based on web-based - decided on a web-based design using React (potentially using Javascript) however library and use case will be expanded on at a later date once the ML pipeline has been finished.
- Server where the model will run off to do FHE and ML inference will either be host-based (localhost) or using AWS/Google/VUW server. This should help with computational power when model performs inference. Will have to evaluate host-based and server-based computations (will be important for final evaluations).
- The plan is to have the UI, evaluation, and ML pipeline finished (by week 6) as this will be the components that will take the most time, but more importantly more error prone. Implementation and integration will be designed in a way where when doing the FHE workflow and pipeline - it can be easily integrated.

## Things to do before next meeting

- Continue with developing the ML pipeline. Ensure that plan, design, and implementation is clearly documented in Gitlab for tracking purposes. Use models developed in 425, which can also help with the evaluation. Having this more or less there will allow, cleaning up, UI and evaluation scripts (which are more or less already implemented) to be cleaned up by Week 6 so the final weeks of the project can be dedicated to the FHE component, evaluation, and finalization.
