from django.contrib.auth.models import AbstractUser

# Get a reference to the AbstractUser class
abstract_user_class = AbstractUser

# Use `dir` to list all attributes (including inherited ones)
attributes = dir(abstract_user_class)

# Print all attributes (might include internal attributes)
print(attributes)

# Filter for attributes starting with underscore (likely inherited)
inherited_attributes = [attr for attr in attributes if attr.startswith("_")]

# Print only inherited attributes
print(inherited_attributes)
