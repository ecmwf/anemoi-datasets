#########################
 Handling missing values
#########################

When handling data for machine learning models, missing values (`NaNs`)
can pose a challenge, as models require complete data to operate
effectively and may crash otherwise. Ideally, we anticipate having
complete data in all fields.

However, there are scenarios where `NaNs` naturally occur, such as with
variables only relevant on land or at sea. This happens for sea surface
temperature (`sst`), for example. In such cases, the default behaviour
is to reject data with `NaNs` as invalid. To accommodate `NaNs` and
accurately compute statistics based on them, you can include the
``allow_nans`` key in the configuration.

Here's an example of how to implement it:

.. literalinclude:: ../yaml/nan.yaml
   :language: yaml
