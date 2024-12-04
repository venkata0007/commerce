from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
from datetime import timedelta



class User(AbstractUser):
    
    pass
def default_end_time():
    return timezone.now() + timedelta(days=7)

class Listing(models.Model):
    FASHION = 'Fashion'
    TOYS = 'Toys'
    ELECTRONICS = 'Electronics'
    HOME = 'Home'
    CHOICES = [
        (FASHION, 'Fashion'),
        (TOYS, 'Toys'),
        (ELECTRONICS, 'Electronics'),
        (HOME, 'Home'),
    ]
    custom_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=64)
    category = models.CharField(max_length=64, choices=CHOICES, default=FASHION)
    description = models.CharField(max_length=64)
    image = models.ImageField(upload_to='media/',default='media/default.jpg')
    starting_bid = models.FloatField()
    # it should be active if current time is between start_time and ending_time 
    active = models.BooleanField(default=False)
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(default=default_end_time)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="listings")
    highest_bid = models.FloatField()

    def __str__(self):
        return f"{self.title} {self.description} {self.category} {self.active} {self.user}"
    
class wishlist(models.Model):
    custom_id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="wishlist")
    listing = models.ForeignKey(Listing, on_delete=models.CASCADE, related_name="wishlist")

    def __str__(self):
        return f"{self.user} {self.listing}"

class Bid(models.Model):
    custom_id = models.AutoField(primary_key=True)
    amount = models.FloatField()
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="bids")
    listing = models.ForeignKey(Listing, on_delete=models.CASCADE, related_name="bids")
    time = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.amount} {self.user} {self.listing} {self.time}"