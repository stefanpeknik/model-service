"use strict";

document.querySelectorAll("[data-quiz]").forEach((quiz) => {
  quiz.querySelectorAll("button[data-correct]").forEach((button) => {
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => {
      quiz.querySelectorAll("button").forEach((choice) => {
        choice.setAttribute("aria-pressed", String(choice === button));
      });
      const correct = button.dataset.correct === "true";
      quiz.dataset.solved = String(correct);
      const feedback = quiz.querySelector(".feedback");
      feedback.dataset.result = correct ? "correct" : "incorrect";
      feedback.textContent = (correct ? "Yes. " : "Try again. ") + button.dataset.feedback;
      const completed = document.querySelectorAll('[data-quiz][data-solved="true"]').length;
      const total = document.querySelectorAll("[data-quiz]").length;
      const progress = document.querySelector("[data-quiz-progress]");
      if (progress) progress.textContent = `${completed} of ${total} predictions currently correct. Explain the final case in your own words to check your reasoning.`;
    });
  });
});
