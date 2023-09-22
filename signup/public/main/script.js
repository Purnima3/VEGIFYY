let user = document.getElementById("user");
user.addEventListener("click", redirect);
function redirect() {
	window.location.href = "http://localhost:3000/";
}

// From locomotive Github > Smooth ke niche
// const scroll = new LocomotiveScroll({
// 	el: document.querySelector(".main"),
// 	smooth: true,
// });

function init() {
	// To make locomotive work with scroll trigger together
	gsap.registerPlugin(ScrollTrigger);

	// Using Locomotive Scroll from Locomotive https://github.com/locomotivemtl/locomotive-scroll

	const locoScroll = new LocomotiveScroll({
		el: document.querySelector(".main"),
		smooth: true,
	});
	// each time Locomotive Scroll updates, tell ScrollTrigger to update too (sync positioning)
	locoScroll.on("scroll", ScrollTrigger.update);

	// tell ScrollTrigger to use these proxy methods for the ".main" element since Locomotive Scroll is hijacking things
	ScrollTrigger.scrollerProxy(".main", {
		scrollTop(value) {
			return arguments.length
				? locoScroll.scrollTo(value, 0, 0)
				: locoScroll.scroll.instance.scroll.y;
		}, // we don't have to define a scrollLeft because we're only scrolling vertically.
		getBoundingClientRect() {
			return {
				top: 0,
				left: 0,
				width: window.innerWidth,
				height: window.innerHeight,
			};
		},
		// LocomotiveScroll handles things completely differently on mobile devices - it doesn't even transform the container at all! So to get the correct behavior and avoid jitters, we should pin things with position: fixed on mobile. We sense it by checking to see if there's a transform applied to the container (the LocomotiveScroll-controlled element).
		pinType: document.querySelector(".main").style.transform
			? "transform"
			: "fixed",
	});
	// each time the window updates, we should refresh ScrollTrigger and then update LocomotiveScroll.
	ScrollTrigger.addEventListener("refresh", () => locoScroll.update());

	// after everything is set up, refresh() ScrollTrigger and update LocomotiveScroll because padding may have been added for pinning, etc.
	ScrollTrigger.refresh();
}
init();

var crsr = document.querySelector(".cursor");
var main = document.querySelector(".main");
main.addEventListener("mousemove", function (dets) {
	// console.log(dets.x);
	crsr.style.left = dets.x + "px";
	crsr.style.top = dets.y + "px";
});

var tl = gsap.timeline({
	scrollTrigger: {
		trigger: ".page1 h1",
		scroller: ".main",
		markers: false,
		start: "top 27%",
		end: "top 0",
		scrub: 3,
	},
});

tl.to(
	".page1 h1",
	{
		x: -100,
	},
	"anim"
);

tl.to(
	".page1 h2",
	{
		x: 100,
	},
	"anim"
);

tl.to(
	".page1 video",
	{
		width: "90%",
	},
	"anim"
);

var tl2 = gsap.timeline({
	scrollTrigger: {
		trigger: ".page1 h1",
		scroller: ".main",
		markers: false,
		start: "top -127%",
		end: "top -130",
		scrub: 3,
	},
});

tl2.to(".page2", {
	backgroundColor: "white",
	color: "black",
});

tl2.to(".page3", {
	backgroundColor: "grey",
	color: "yellow",
});

tl2.to(".page4", {
	backgroundColor: "#FFFFED",
	color: "black",
});

var searchbtn1 = document.querySelector(".search-btn");
var searchInput1 = document.querySelector(".search-Input");

function filterProduct1() {
	const searchVal = searchInput1.value.toLowerCase();
	const prodItems = document.querySelectorAll(".ccontent");

	prodItems.forEach((item) => {
		const title = item.querySelector("h3").innerText.toLowerCase();
		console.log("Prod1");
		if (title.includes(searchVal) || searchVal == "") {
			item.style.display = "block";
		} else {
			item.style.display = "none";
		}
	});
}
searchbtn1.addEventListener("click", filterProduct1);
searchInput1.addEventListener("keyup", filterProduct1);
filterProduct1();

// For the second set of search input and button
var searchbtn2 = document.querySelector(".search-btn-2");
var searchInput2 = document.querySelector(".search-Input-2");

function filterProduct2() {
	const searchVal = searchInput2.value.toLowerCase();
	const prodItems = document.querySelectorAll(".ccontent");

	prodItems.forEach((item) => {
		const title = item.querySelector("h3").innerText.toLowerCase();

		if (title.includes(searchVal) || searchVal === "") {
			item.style.display = "block";
		} else {
			item.style.display = "none";
		}
	});
}

searchbtn2.addEventListener("click", filterProduct2);
searchInput2.addEventListener("keyup", filterProduct2);
filterProduct2();

// For the third set of search input and button
var searchbtn3 = document.querySelector(".search-btn-3");
var searchInput3 = document.querySelector(".search-Input-3");

function filterProduct3() {
	const searchVal = searchInput3.value.toLowerCase();
	const prodItems = document.querySelectorAll(".ccontent");

	prodItems.forEach((item) => {
		const title = item.querySelector("h3").innerText.toLowerCase();

		if (title.includes(searchVal) || searchVal === "") {
			item.style.display = "block";
		} else {
			item.style.display = "none";
		}
	});
}

searchbtn3.addEventListener("click", filterProduct3);
searchInput3.addEventListener("keyup", filterProduct3);
filterProduct3();
