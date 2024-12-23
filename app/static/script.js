// A basic script to handle button clicks
document.addEventListener("DOMContentLoaded", function () {
    const buttons = document.querySelectorAll("button");

    buttons.forEach(button => {
        button.addEventListener("click", () => {
            alert("Button clicked!");
        });
    });
});
