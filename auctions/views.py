from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.urls import reverse

from django.contrib.auth.decorators import login_required

from .models import *

from django.contrib import messages
def index(request):
    return render(request, "auctions/index.html")


def login_view(request):
    if request.method == "POST":

        # Attempt to sign user in
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)

        # Check if authentication successful
        if user is not None:
            login(request, user)
            return HttpResponseRedirect(reverse("index"))
        else:
            return render(request, "auctions/login.html", {
                "message": "Invalid username and/or password."
            })
    else:
        return render(request, "auctions/login.html")


def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse("index"))


def register(request):
    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]

        # Ensure password matches confirmation
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]
        if password != confirmation:
            return render(request, "auctions/register.html", {
                "message": "Passwords must match."
            })

        # Attempt to create new user
        try:
            user = User.objects.create_user(username, email, password)
            user.save()
        except IntegrityError:
            return render(request, "auctions/register.html", {
                "message": "Username already taken."
            })
        login(request, user)
        return HttpResponseRedirect(reverse("index"))
    else:
        return render(request, "auctions/register.html")
def listing(request, listing_id = None):
    if listing_id is not None:
        listings = Listing.objects.get(custom_id =listing_id)
        bids = Bid.objects.filter(listing = listings)
        return render(request, "auctions/display.html", {
            "object": listings,
            "bids": bids
        })
    else:
        listings = Listing.objects.all() #.filter(active=True)
        return render(request, "auctions/listing.html", {
            "listings": listings
        })
        
@login_required
def create(request):
    if request.method == "POST":
        title = request.POST["title"]
        description = request.POST["description"]
        # starting_bid = request.POST["starting_bid"]
        image_url = request.POST["image"]
        print(image_url)
        category = request.POST["category"]
        # user = request.POST["user"]
        bid = request.POST['price']
        user = request.user
        print(user)
        try:
            listing = Listing.objects.create(title=title, description=description, image=image_url,starting_bid = bid, category=category, user=user,highest_bid = bid)#, start_time=start_time, ending_time=ending_time)
            listing.save()
            return HttpResponseRedirect(reverse("index"))
        except Exception as e:
            print(e)
            return render(request, "auctions/create.html", {
                "message": "Listing already exists."
            })
    else:
        return render(request, "auctions/create.html")
@login_required
def my_listings(request):
    
    print(request.user)
    user = request.user
    # print(user.user_id)
    listings = Listing.objects.filter(user=user)  # Filter listings by user
    return render(request, "auctions/listing.html", {
        "listings": listings,
    })


@login_required
def update_listing(request,listing_id):
    user_id = Listing.objects.get(custom_id=listing_id).user.id
    if(request.user.id!= user_id):
        return HttpResponse("You are not authorized to update this product")
    product = Listing.objects.get(custom_id=listing_id)
    if request.method == "GET":
        return render(request, "auctions/update_listing.html", {
            'listing_id': listing_id
        })
    
    if request.method == "POST":
        product.title = request.POST["title"]
        product.description = request.POST["description"]
        product.starting_bid = request.POST["starting_bid"]
        product.image = request.POST["image"]
        product.category = request.POST["category"]
        product.active = request.POST["active"]
        product.highest_bid = request.POST["starting_bid"]
        product.save()
    return HttpResponseRedirect(reverse("display"),listing = product)

@login_required
def bid(request, listing_id):
    if request.method == "POST":
        bid = request.POST["bid"]
        listing = Listing.objects.get(custom_id=listing_id)
        #if bid is before end time
        if listing.end_time < timezone.now():
            return render(request, "auctions/display.html", {
                "object": listing,
                "message": "Auction has ended.",
                "is_a_bid": False
            })
        if float(bid) > listing.highest_bid:
            listing.highest_bid = float(bid)
            listing.save()
            current_bid = Bid.objects.create(amount=bid, user=request.user, listing=listing)
            current_bid.save()
            return HttpResponseRedirect(reverse("listing:{listing.custom_id}"),)
            # is_a_bid = True
            # return render(request, "auctions/display.html", {
            #     "object": listing,
            #     "message": "Bid must be higher than current bid.",
            #     "is_a_bid": is_a_bid
            # })
        else:
            is_a_bid = True
            return render(request, "auctions/display.html", {
                "object": listing,
                "message": "Bid must be higher than current bid.",
                "is_a_bid": is_a_bid
            })
    else:
        object = Listing.objects.get(custom_id=listing_id)
        return render(request, "auctions/bid.html", {
        "object": object,
    })

