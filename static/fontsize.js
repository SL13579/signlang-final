// 글씨 크기 변경 함수
function setFontSize(size) {
    if (size === 'large') {
      document.documentElement.setAttribute("data-fontsize", "large");
      localStorage.setItem("fontSize", "large");
    } else {
      document.documentElement.removeAttribute("data-fontsize");
      localStorage.setItem("fontSize", "normal");
    }
  }
  
  // 페이지 로드 시 저장된 설정 적용
  document.addEventListener("DOMContentLoaded", function () {
    const saved = localStorage.getItem("fontSize");
    if (saved === "large") {
      document.documentElement.setAttribute("data-fontsize", "large");
    }
  });
  