from dataset import Dataset, DatasetEntry

class MockUpDataset(Dataset):
    async def generate_context(self)->DatasetEntry:
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>Tech Company</title>
<style>
    body {
    margin: 0;
    font-family: Arial, sans-serif;
    color: #fff;
    background-color: #1a2632;
}

.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    background-color: #1a2632;
}

.header-container .logo {
    font-size: 24px;
}

nav ul {
    list-style: none;
    display: flex;
    gap: 20px;
}

nav ul li a {
    color: #fff;
    text-decoration: none;
}

.hero {
    width: 100%;
    height: auto;
}

.hero img {
    width: 100%;
    height: auto;
}

.welcome {
    text-align: center;
    padding: 40px;
    background-color: #1a2632;
}

.welcome h1 {
    font-size: 36px;
    margin-bottom: 20px;
}

.welcome p {
    font-size: 16px;
    line-height: 1.5;
    max-width: 600px;
    margin: 0 auto;
}

footer {
    padding: 20px;
    background-color: #28323c;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

footer p {
    margin: 0;
}

.footer-container .socials a {
    color: #fff;
    text-decoration: none;
    margin-left: 20px;
}
</style>
<link href="styles.css" rel="stylesheet"/>
</head>
<body>
<header>
<div class="header-container">
<div class="logo">Tech Company</div>
<nav>
<ul>
<li><a href="#">Home</a></li>
<li><a href="#">About</a></li>
<li><a href="#">Products</a></li>
<li><a href="#">Contact</a></li>
</ul>
</nav>
</div>
</header>
<section class="hero">
<img alt="Hero Image" src="https://picsum.photos/seed/picsum/800/600"/>
</section>
<section class="welcome">
<h1>Welcome to Tech Company</h1>
<p>At Tech Company, we are dedicated to providing the best technology solutions for your needs. Our team of experts is always ready to help you with any questions or problems you may have.</p>
</section>
<footer>
<div class="footer-container">
<p>© 2022 Tech Company. All rights reserved.</p>
<div class="socials">
<a href="#">Facebook</a>
<a href="#">Twitter</a>
<a href="#">Instagram</a>
</div>
</div>
</footer>
</body>
</html>
        """
        return DatasetEntry(
            src="mockup",
            topic="tech company",
            ground_truth_html=html,
            prompt="",
            base64_image=""
        )

class MockUpPromptDataset(Dataset):
    async def generate_context(self)->DatasetEntry:
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coming Soon</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4;
            color: #333;
            text-align: center;
        }

        /* Header styles */
        header {
            background-color: #333;
            color: white;
            padding: 10px 0;
        }

        header nav {
            display: flex;
            justify-content: center;
        }

        header nav a {
            color: white;
            margin: 0 15px;
            text-decoration: none;
            font-weight: bold;
        }

        header nav a:hover {
            text-decoration: underline;
        }

        /* Coming Soon section styles */
        .coming-soon {
            margin-top: 50px;
        }

        .coming-soon h1 {
            font-size: 2.5em;
            color: #333;
        }

        .coming-soon p {
            font-size: 1.2em;
            color: #555;
        }

        /* Go back button styles */
        .go-back-btn {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            font-size: 1.1em;
            cursor: pointer;
            margin-top: 20px;
        }

        .go-back-btn:hover {
            background-color: #0056b3;
        }

        /* Footer styles */
        footer {
            background-color: #333;
            color: white;
            padding: 20px;
            position: fixed;
            width: 100%;
            bottom: 0;
        }

        footer p {
            margin: 0;
            font-size: 1em;
        }

        footer a {
            color: #00bcd4;
            text-decoration: none;
        }

        footer a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>

    <!-- Header with Navigation -->
    <header>
        <nav>
            <a href="#">Home</a>
            <a href="#">About</a>
            <a href="#">Contact</a>
        </nav>
    </header>

    <!-- Coming Soon Section -->
    <div class="coming-soon">
        <h1>Coming Soon!</h1>
        <p>We're working hard to launch something amazing. Stay tuned!</p>
        <button class="go-back-btn" onclick="window.history.back();">Go Back</button>
    </div>

    <!-- Footer -->
    <footer>
        <p>&copy; 2024 Your Website | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></p>
    </footer>

</body>
</html>

"""
        return DatasetEntry(
            src="mockup",
            topic="tech company",
            ground_truth_html=html,
            prompt="CommingSoon Page with goback button, navHeader, and footer",
            base64_image=""
        )