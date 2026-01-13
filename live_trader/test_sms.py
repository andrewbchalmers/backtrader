from pushbullet import Pushbullet

# Get your API key from: https://www.pushbullet.com/#settings/account
# 1. Go to pushbullet.com and sign up (free)
# 2. Install the app on your phone
# 3. Go to Settings → Account → Create Access Token
PUSHBULLET_API_KEY = "o.ptYJ8W8YpFEnDVZ1CL4vO9N7suOvJURG"

pb = Pushbullet(PUSHBULLET_API_KEY)

# Send test notification
#push = pb.push_note("Trading Alert Test", "BUY NVDA @ $184.86\nSL @ $166.37")
push = pb.push_note("Trading Alert Test", "NVDA REGISTERED AS BOUGHT")

print("✓ Notification sent!")
print(f"Push ID: {push['iden']}")