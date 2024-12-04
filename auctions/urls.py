from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.index, name="index"),
    path("login", views.login_view, name="login"),
    path("logout", views.logout_view, name="logout"),
    path("register", views.register, name="register"),
    path("create", views.create, name="create"),
    path("listing", views.listing, name="all_listings"),
    path("listing/<int:listing_id>", views.listing, name="listing"),
    path("my_listings", views.my_listings, name="my_listings"),
    path("update_listing/<int:listing_id>", views.update_listing, name="update_listing"),
    path("bid/<int:listing_id>", views.bid, name="bid"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)