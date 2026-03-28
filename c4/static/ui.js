document.addEventListener("DOMContentLoaded", () => {
  const revealTargets = document.querySelectorAll(
    ".hero-card, .panel-card, .feature-card, .table-responsive"
  );

  revealTargets.forEach((el, idx) => {
    el.classList.add("reveal");
    if (idx % 3 === 1) el.classList.add("reveal-delay-1");
    if (idx % 3 === 2) el.classList.add("reveal-delay-2");
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -30px 0px" }
  );

  revealTargets.forEach((el) => observer.observe(el));
});

