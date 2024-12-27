// Check localStorage for saved theme preference on page load
document.addEventListener('DOMContentLoaded', () => {
    const toggleButton = document.getElementById('toggleTheme');

    // Apply the saved theme from localStorage if it exists
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-mode');
    }

    // Toggle theme and save preference to localStorage
    toggleButton.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        // Save the theme preference to localStorage
        localStorage.setItem('theme', document.body.classList.contains('dark-mode') ? 'dark' : 'light');
        toggleButton.textContent = document.body.classList.contains('dark-mode') ? '☀️' : '🌙';
    });
});
